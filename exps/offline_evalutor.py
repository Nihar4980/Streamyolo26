# ==================================================
# Third-Party Libraries
# ==================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils import LOGGER, RANK, nms, ops
import torch.distributed as dist


import contextlib
import io
import itertools
import json
import tempfile
import time

from tabulate import tabulate
from faster_coco_eval import COCO, COCOeval_faster

import time
import torch

def time_synchronized():
    """Return synchronized wall-clock time for CPU/GPU operations."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()
def latency_stats(values):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }




ARGOVERSE_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
)

def per_class_mAP_table(coco_eval, class_names=ARGOVERSE_CLASSES, headers=["class", "AP"], colums=2):
    per_class_mAP = {}
    precisions = coco_eval.eval["precision"]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_mAP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_mAP) * len(headers))
    result_pair = [x for pair in per_class_mAP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table
class DetectionValidator:
    def __init__(
        self,
        model,
        val_loader,
        device,
    ):
        self.model = model
        self.val_loader = val_loader
        self.device = device


    def evaluate(self):
        self.model.eval()
        disable = (
                        dist.is_available()
                        and dist.is_initialized()
                        and dist.get_rank() != 0
                  )
                    
        pbar = tqdm(
                        enumerate(self.val_loader),
                        total=len(self.val_loader),
                        leave=False,
                        disable=disable,
                    )
        inference_time = 0
        nms_time = 0
        n_samples = len(self.val_loader) - 1
        self.tloss = None
        data_list = []

        forward_latencies = []  # milliseconds per image
        nms_latencies = []      # milliseconds per image
        total_latencies = []    # milliseconds per image
        
        warmup_batches = 20
        for i, batch in pbar:
            # ---------------------------
            # Validate batch tensors
            # ---------------------------
            batch["img"] = batch["img"].to(self.device, non_blocking=True).float().div_(255.0).half()
    
            for k in ("cls_curr","bboxes_curr","batch_idx_curr", "cls_next","bboxes_next","batch_idx_next"):
                batch[k] = batch[k].to(self.device, non_blocking=True)
            img_ids = batch["img_id"]
            with torch.no_grad():

                # Synchronize before starting the forward-pass timer
                torch.cuda.synchronize(self.device)
                start = time.perf_counter()
            
                preds, raw_preds = self.model(batch["img"], mode="off_pipe")
            
                torch.cuda.synchronize(self.device)
                infer_end = time.perf_counter()
            
                # Batch forward time
                forward_batch_ms = (infer_end - start) * 1000.0
                batch_size = len(preds)
            
                # Exclude initial batches from timing statistics
                record_timing =  i < len(self.val_loader)
            
                # Average forward latency per image within this batch
                forward_per_image_ms = forward_batch_ms / max(batch_size, 1)
        
            for b in range(batch_size):
        
                pred = preds[b]
                img_id = int(img_ids[b])
        
                # Measure NMS and post-processing for this image
                torch.cuda.synchronize(self.device)
                nms_start = time.perf_counter()
        
                bboxes, conf, cls, _ = self.inference(
                    pred.unsqueeze(0)
                )
        
                torch.cuda.synchronize(self.device)
                nms_end = time.perf_counter()
        
                nms_per_image_ms = (nms_end - nms_start) * 1000.0
                total_per_image_ms = (
                    forward_per_image_ms + nms_per_image_ms
                )
        
                if record_timing:
                    forward_latencies.append(forward_per_image_ms)
                    nms_latencies.append(nms_per_image_ms)
                    total_latencies.append(total_per_image_ms)
        
                # Keep your existing COCO detection format
                for j in range(len(bboxes)):
                    data_list.append({
                        "image_id": img_id,
                        "bbox": bboxes[j],
                        "score": conf[j],
                        "category_id": cls[j],
                    })
            
                    #data_list.append(result_dict)

        forward_stats = latency_stats(forward_latencies)
        nms_stats = latency_stats(nms_latencies)
        total_stats = latency_stats(total_latencies)
        
        # Approximate FPS based on mean per-image total latency
        fps = (
            1000.0 / total_stats["mean"]
            if np.isfinite(total_stats["mean"]) and total_stats["mean"] > 0
            else float("nan")
        )
        
        print("\n========== LATENCY STATISTICS ==========")
        print(f"Measured images: {len(total_latencies)}")
        
        for name, stats in [
            ("Forward", forward_stats),
            ("NMS + post-processing", nms_stats),
            ("Total", total_stats),
        ]:
            print(
                f"{name:24s} | "
                f"Mean: {stats['mean']:.3f} ms | "
                f"STD: {stats['std']:.3f} ms | "
                f"Min: {stats['min']:.3f} ms | "
                f"Max: {stats['max']:.3f} ms"
            )
        
        print(f"Approximate FPS: {fps:.2f}")
        print("========================================\n")

        # statistics = torch.cuda.FloatTensor([nms_per_image_ms, n_samples])

        # inference_time = statistics[0].item()
        # nms_time = statistics[1].item()
        # n_samples = statistics[2].item()
        # a_infer_time = 1000 * inference_time / (n_samples * self.val_loader.batch_size)
        # a_nms_time = 1000 * nms_time / (n_samples * self.val_loader.batch_size)

        # time_info = ", ".join(
        #     [
        #         "Average {} time: {:.2f} ms".format(k, v)
        #         for k, v in zip(
        #         ["forward", "NMS", "inference"],
        #         [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
        #     )
        #     ]
        # )

        # info = time_info + "\n"

        if len(data_list) > 0:
            cocoGt = COCO("/home/nihar22951/Streamyolo26/Datasets/Argoverse-HD/annotations/val.json")
            _, tmp = tempfile.mkstemp()
            json.dump(data_list, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval_faster(
                                        cocoGt, cocoDt, "bbox",
                                        print_function=print
                                      )

            cocoEval.evaluate()
            cocoEval.accumulate()
            # redirect_string = io.StringIO()
            # with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
            # info += redirect_string.getvalue()
            # info += "per class mAP:\n" + per_class_mAP_table(cocoEval)
            class_map =  per_class_mAP_table(cocoEval)
            print(class_map)
        else:
            return 0
        
               
                
    def inference(  self,
                    preds,
                    img_shape=(600, 960),        # (H, W) model input
                    orig_shape=(1200, 1920),     # (H, W) original image
                ):
                    """
                    Ultralytics-style inference.
                    Returns:
                        bboxes : COCO XYWH (top-left)
                        scores : confidence
                        labels : class ids
                        extra  : None
                    """
                
                    outputs = nms.non_max_suppression(
                        prediction=preds,
                        conf_thres=0.01,
                        iou_thres=0.65,
                        classes=None,
                        agnostic=False,
                        multi_label=True,
                        max_det=300,
                        nc=8,
                    )
                
                    if len(outputs) == 0:
                        return (
                            np.empty((0, 4), dtype=np.float32),
                            np.empty((0,), dtype=np.float32),
                            np.empty((0,), dtype=np.int32),
                            None,
                        )
                
                    # Keep on torch until all processing is complete
                    outputs = outputs[0]
                    outputs = outputs.cpu()
                
                    # ------------------------------------------------------------------
                    # Ultralytics: keep XYXY after NMS
                    # ------------------------------------------------------------------
                    boxes = outputs[:, :4].clone()
                
                    # ------------------------------------------------------------------
                    # Ultralytics-style scaling
                    # ------------------------------------------------------------------
                    boxes = ops.scale_boxes(
                        img_shape,      # network input (H, W)
                        boxes,
                        orig_shape      # original image (H, W)
                    )
                
                    # ------------------------------------------------------------------
                    # XYXY -> XYWH (center)
                    # ------------------------------------------------------------------
                    boxes = ops.xyxy2xywh(boxes)
                
                    # ------------------------------------------------------------------
                    # Center XYWH -> top-left XYWH (COCO)
                    # ------------------------------------------------------------------
                    boxes[:, :2] -= boxes[:, 2:] / 2
                
                    return (
                        boxes.numpy().astype(np.float32).tolist(),
                        outputs[:, 4].numpy().astype(np.float32).tolist(),
                        outputs[:, 5].numpy().astype(np.int32).tolist(),
                        None,
                    )


            
   

  
  