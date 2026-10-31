# ==========================================================
# 1️⃣ Imports
# ==========================================================
import os
import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from loss import v8DetectionLoss
from dataset import *
from model import *
from trainer import *
import time
from pathlib import Path




# ==========================================================
# 2️⃣ DDP Setup
# ==========================================================
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def random_samples(dataset, n=10):
    indices = np.random.choice(len(dataset), size=n, replace=False)
    return Subset(dataset, indices), indices

class_names = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "bus",
        5: "truck",
        6: "traffic_light",
        7: "stop_sign",
    }

# ==========================================================
# 5️⃣ Main
# ==========================================================
def main(model, image_train, label_train, image_val, label_val,flip= None, freeze=None,batch_size=8,epochs=30):
    if freeze is None:
        freeze = []

    is_ddp, rank, local_rank, world_size = setup_ddp()

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    torch.backends.cudnn.benchmark = True
    model = model.to(device) 
    if is_ddp:
        model = DDP(model, device_ids=[local_rank],find_unused_parameters=True,)

    if flip:
        from torch.utils.data import ConcatDataset
    
        train_dataset = AgroverseDataset(
            image_train,
            label_train,
            box_transform,
            image_transform,
        )
        
        train_dataset_flip = AgroverseDataset(
            image_train,
            label_train,
            box_transform_flip,
            image_transform_flip,
        )
        
        train_dataset = ConcatDataset([
            train_dataset,
            train_dataset_flip
        ])
    else:
        train_dataset = AgroverseDataset(
            image_train,
            label_train,
            box_transform,
            image_transform,
        )
        

    train_subset, indices = random_samples(train_dataset, n=1)

    train_sampler = DistributedSampler(train_dataset) if is_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_dataset = AgroverseDataset(
        image_val,
        label_val,
        box_transform,
        image_transform
    )

    val_subset, indices = random_samples(val_dataset, n=1)

    val_sampler = DistributedSampler(val_dataset) if is_ddp else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True
    )


    loss = v8DetectionLoss(device=device, reg_max=1)

    trainer = DetectionTrainer(
                                    model=model,
                                    train_loader= train_loader,
                                    loss=loss,
                                    device=device,
                                    val_loader=val_loader,
                                    class_names=class_names,
                                    epochs= epochs,
                                    optimizer_name="MuSGD",
                                    batch_size = batch_size,
                                    freeze = freeze,
                                    path = "/home/nihar22951/Streamyolo26/weights/Streamyolo26s_attn"
                                    
                              )

    trainer.train(rank=rank, sampler=train_sampler)

    cleanup_ddp()


def load_pretrained_weights(model, pretrained, prefix_map, strict=False, verbose=True):
    """
    Load matching pretrained weights into a custom model.

    Args:
        model: Custom PyTorch model.
        pretrained: Pretrained PyTorch model.
        prefix_map: Dict mapping custom module prefixes to pretrained prefixes.
        strict: Passed to load_state_dict().
        verbose: Print copied and skipped parameters.

    Returns:
        model: Updated model.
        mapping: Dictionary of matched parameter names.
    """

    my_sd = model.state_dict()
    pre_sd = pretrained.state_dict()

    mapping = {}
    copied = 0
    matched = []
    shape_mismatch = []
    missing = []

    for dst in my_sd.keys():
        for my_prefix, pre_prefix in prefix_map.items():

            if dst.startswith(my_prefix + "."):
                src = dst.replace(my_prefix, pre_prefix, 1)

                if src in pre_sd:
                    if pre_sd[src].shape == my_sd[dst].shape:
                        mapping[dst] = src
                        matched.append(dst)
                        my_sd[dst] = pre_sd[src]
                        copied += 1
                    elif verbose:
                        shape_mismatch.append(dst)
                        print(f"Shape mismatch:")
                        print(f"  {dst} {my_sd[dst].shape}")
                        print(f"  {src} {pre_sd[src].shape}")
                elif verbose:
                    missing.append(dst)
                    print(f"Missing pretrained key: {src}")

                break

    msg = model.load_state_dict(my_sd, strict=strict)

    if verbose:
        print(f"\nCopied {copied}/{len(mapping)} tensors.")
        print(f"Missing keys     : {len(msg.missing_keys)}")
        print(f"Unexpected keys  : {len(msg.unexpected_keys)}")

    return model, matched, shape_mismatch, missing


# ==========================================================
# 6️⃣ Run
# ==========================================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    image_train = "/home/nihar22951/Streamyolo26/Datasets/Argoverse-1.1/images/train"
    label_train = "/home/nihar22951/Streamyolo26/Datasets/Argoverse-1.1/labels/train"
    # Val paths
    image_val = "/home/nihar22951/Streamyolo26/Datasets/Argoverse-1.1/images/val"
    label_val = "/home/nihar22951/Streamyolo26/Datasets/Argoverse-1.1/labels/val"
    
    from ultralytics import YOLO

    #model = yolon(scale="nano", reg_max=1, nc=8, improver=True)
    model = yolos(scale="small", reg_max=1, nc=8, global_attn = True)
    #model = yolom(scale="medium",reg_max=1, nc=8, improver=True)
    #model = yolol(reg_max=1, nc=8, improver=True)
    #model = yolox(reg_max=1, nc=8, improver=True)
    model.train()
    model.detect.bias_init()
    
    pretrained = YOLO("/home/nihar22951/Streamyolo26/Weights/yolo26/yolo26s.pt").model
    #pretrained11 = YOLO("/home/nihar22951/Streamyolo26/Weights/yolo11/yolo11s.pt").model
    
    prefix_map = {
                    "conv1": "model.0",
                    "conv2": "model.1",
                    "c3_1": "model.2",
                    "conv3": "model.3",
                    "c3_2": "model.4",
                    "conv4": "model.5",
                    "c3_3": "model.6",
                    "conv5": "model.7",
                    "c3_4": "model.8",
                    "sppf": "model.9",
                    "c2psa": "model.10",
                    "fuse1": "model.13",
                    "fuse2": "model.16",
                    "down1": "model.17",
                    "fuse3": "model.19",
                    "down2": "model.20",
                    "fuse4": "model.22",
                    "detect":"model.23"
                }
    #prefix_map11= {"detect":"model.23"}
    
    model, matched, mismatched, missing = load_pretrained_weights(
                                                                    model,
                                                                    pretrained,
                                                                    prefix_map,
                                                                    strict=False,
                                                                    verbose=True,
                                                                )
    # model, _,_,_ = load_pretrained_weights(
    #                                                                 model,
    #                                                                 pretrained11,
    #                                                                 prefix_map11,
    #                                                                 strict=False,
    #                                                                 verbose=True,
    #                                                             )
    
    # batch_size=4
    # epochs=5

    main(model, 
         image_train= image_train, 
         label_train=label_train, 
         image_val= image_val, 
         label_val= label_val,
         flip=True,
         batch_size=8,
         epochs=15,
         #freeze= matched
        )
 