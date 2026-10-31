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


class DetectionValidator:
    def __init__(
        self,
        model,
        val_loader,
        loss,
        device,
        class_names,
        training = True,
        prefix="val"
    ):
        self.model = model
        self.val_loader = val_loader
        self.criterion = loss
        self.device = device

        self.metrics = DetMetrics()
        self.confusion_matrix = ConfusionMatrix(
            names=class_names,
            task="detect",
        )
        self.training = training
        self.prefix = prefix

    def validate(self):
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
        self.tloss = None
        for i, batch in pbar:
            # ---------------------------
            # Validate batch tensors
            # ---------------------------
            batch["img"] = batch["img"].to(self.device, non_blocking=True).float().div_(255.0)
    
            for k in ("cls_curr","bboxes_curr","batch_idx_curr", "cls_next","bboxes_next","batch_idx_next"):
                batch[k] = batch[k].to(self.device, non_blocking=True)

            with torch.no_grad():
                preds, raw_preds = self.model(batch['img'], mode='off_pipe')
                if self.training and self.prefix=="val":
                    self.loss_items = self.criterion(raw_preds, batch)[1]

                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                

            preds = self.postprocess(preds)

            for si, pred in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)

                cls = pbatch["cls"].cpu().numpy()
                no_pred = pred["cls"].shape[0] == 0

                self.metrics.update_stats(
                    {
                        **self._process_batch(pred, pbatch),
                        "target_cls": cls,
                        "target_img": np.unique(cls),
                        "conf": (
                            np.zeros(0)
                            if no_pred
                            else pred["conf"].cpu().numpy()
                        ),
                        "pred_cls": (
                            np.zeros(0)
                            if no_pred
                            else pred["cls"].cpu().numpy()
                        ),
                    }
                )
                if not self.training:
                    self.confusion_matrix.process_batch(pred, pbatch)

        stats = self.get_stats()
        
        if self.training and self.prefix=="val":
             if dist.is_available() and dist.is_initialized():
                    tloss = self.tloss.clone()
                    dist.all_reduce(tloss, op=dist.ReduceOp.SUM)
                    tloss /= dist.get_world_size()
                    self.tloss = tloss
             loss  = {
                    "box_loss": self.tloss[0].cpu(),
                    "cls_loss": self.tloss[1].cpu(),
                    "dfl_loss": self.tloss[2].cpu(),
                   }
             results = {**stats,**loss}
            
             return {f"{self.prefix}/{k}": round(float(v), 5) for k, v in results.items()}
        else:
            return stats
            
    def get_stats(self):
        if dist.is_available() and dist.is_initialized():
            self.gather_stats()
    
        self.metrics.process()
        results = self.metrics.results_dict
        self.metrics.clear_stats()
    
        return results

    def gather_stats(self):
        """Gather validation statistics from all ranks."""
        if not (dist.is_available() and dist.is_initialized()):
            return
    
        world_size = dist.get_world_size()
    
        gathered_stats = [None] * world_size
        dist.all_gather_object(gathered_stats, self.metrics.stats)
    
        merged_stats = {k: [] for k in self.metrics.stats.keys()}
    
        for stats in gathered_stats:
            for k in merged_stats:
                merged_stats[k].extend(stats[k])
    
        self.metrics.stats = merged_stats

    def postprocess(self, preds: torch.Tensor):
        """
        Apply Non-Maximum Suppression to prediction outputs.
        """
        outputs = nms.non_max_suppression(
            prediction=preds,
            conf_thres=0.05,
            iou_thres=0.5,
            nc=8,
        )

        return [
            {
                "bboxes": x[:, :4],
                "conf": x[:, 4],
                "cls": x[:, 5],
                "extra": x[:, 6:],
            }
            for x in outputs
        ]

    def _prepare_batch(self, si: int, batch: dict):
        """
        Prepare a batch of images and annotations for validation.
        """
        idx = batch["batch_idx_next"] == si
        cls = batch["cls_next"][idx].squeeze(-1)
        bbox = batch["bboxes_next"][idx]

        imgsz = batch["img"].shape[2:]

        if cls.shape[0]:
            bbox = (
                ops.xywh2xyxy(bbox)
                * torch.tensor(
                    imgsz,
                    device=bbox.device,
                )[[1, 0, 1, 0]]
            )

        return {
            "cls": cls,
            "bboxes": bbox,
            "imgsz": imgsz,
        }

    @staticmethod
    def match_predictions(
        pred_classes: torch.Tensor,
        true_classes: torch.Tensor,
        iou: torch.Tensor,
        use_scipy: bool = False,
    ):
        """
        Match predictions to ground truth objects using IoU.
        """
        iouv = torch.linspace(0.5, 0.95, 10)

        correct = np.zeros(
            (pred_classes.shape[0], iouv.shape[0])
        ).astype(bool)

        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class
        iou = iou.cpu().numpy()

        for i, threshold in enumerate(iouv.cpu().tolist()):
            if use_scipy:
                import scipy

                cost_matrix = iou * (iou >= threshold)

                if cost_matrix.any():
                    labels_idx, detections_idx = (
                        scipy.optimize.linear_sum_assignment(cost_matrix)
                    )

                    valid = (
                        cost_matrix[
                            labels_idx,
                            detections_idx,
                        ]
                        > 0
                    )

                    if valid.any():
                        correct[detections_idx[valid], i] = True

            else:
                matches = np.nonzero(iou >= threshold)
                matches = np.array(matches).T

                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[
                            iou[
                                matches[:, 0],
                                matches[:, 1],
                            ].argsort()[::-1]
                        ]
                        matches = matches[
                            np.unique(
                                matches[:, 1],
                                return_index=True,
                            )[1]
                        ]
                        matches = matches[
                            np.unique(
                                matches[:, 0],
                                return_index=True,
                            )[1]
                        ]

                    correct[
                        matches[:, 1].astype(int),
                        i,
                    ] = True

        return torch.tensor(
            correct,
            dtype=torch.bool,
            device=pred_classes.device,
        )

    def _process_batch(self, preds: dict, batch: dict):
        """
        Return correct prediction matrix.
        """
        iouv = torch.linspace(0.5, 0.95, 10)
        niou = iouv.numel()

        if (
            batch["cls"].shape[0] == 0
            or preds["cls"].shape[0] == 0
        ):
            return {
                "tp": np.zeros(
                    (preds["cls"].shape[0], niou),
                    dtype=bool,
                )
            }

        iou = box_iou(
            batch["bboxes"],
            preds["bboxes"],
        )

        return {
            "tp": self.match_predictions(
                preds["cls"],
                batch["cls"],
                iou,
            )
            .cpu()
            .numpy()
        }