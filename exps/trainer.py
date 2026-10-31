import gc
import time
import logging
import os
import platform
import sys
import torch
import math
import random
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import pandas as pd
from torch import optim
from functools import partial
from ultralytics.optim import MuSGD

from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model)
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import torch.distributed as dist
from ultralytics.utils.plotting import plot_results
from pathlib import Path
#from ultralytics.utils.torch_utils import EarlyStopping



from validator import *




class DetectionTrainer:
    def __init__(
        self,
        model,
        train_loader,
        loss,
        device,
        val_loader,
        class_names,
        path,
        freeze = [],
        epochs=50,
        # patience=15,
        amp=True,
        input_size = (600,960),
        multiscale_range = 5,
        multi_scale = True,

        # Optimizer params
        optimizer_name="auto",
        lr0=0.00038,
        momentum=0.94751,
        weight_decay=0.00027,

        # Warmup params
        warmup_epochs=0.98745,
        warmup_momentum=0.54064,
        warmup_bias_lr=0.05684,

        # LR schedule
        cos_lr=True,
        lrf=0.88219,
        nbs=64,
        nc=8,
        batch_size=16, 
    ):
        from pathlib import Path

        self.save_dir = Path(path)
        #self.wdir.mkdir(parents=True, exist_ok=True)
        self.wdir = self.save_dir / "weights"  # weights dir
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.csv = self.save_dir / "result.csv"
        
        # self.ckpt_path = self.wdir / file
        
        self.device = device
        self.epochs = epochs
        self.train_loader = train_loader 
        #self.use_amp = use_amp

        # ---------------------------
        # Model, EMA, Loss
        # ---------------------------
        self.model = model.to(device)
        self.criterion = loss
        
        # ---------------------------
        # Training params
        # ---------------------------
        self.nc = nc
        self.warmup_epochs = warmup_epochs
        self.cos_lr = cos_lr
        self.lr0 = lr0
        self.lrf = lrf
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.nbs = nbs
        self.batch_size = batch_size
        self.freeze = freeze
        self.optimizer_name = optimizer_name 
        self.nan_recovery_attempts = 0
        self.input_size = input_size
        self.multiscale_range = multiscale_range
        self.multi_scale = multi_scale
                                                  

         # Validator
        self.amp = amp
        self.val_loader = val_loader
        self.class_names = class_names
        self.metrics = None
        self.best_fitness = None
        self.fitness = None
        self.plots = {}
        # self.patience = patience
        # self.stopper = EarlyStopping(patience=self.patience)
        
       
    def _setup_train(self):
        """Configure model, optimizer, dataloaders, and training utilities before the training loop."""
        
        self.model = self.model.to(self.device)
    
        #self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)
        
    
        # Freeze layers
        freeze_list = (
            self.freeze
            if isinstance(self.freeze, list)
            else range(self.freeze)
            if isinstance(self.freeze, int)
            else []
        )
        
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [x for x in freeze_list] + always_freeze_names # f"module.{x}"
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True
        if not any(v.requires_grad for v in self.model.parameters()):
            raise RuntimeError(
                f"'freeze={self.args.freeze}' froze the entire model with no trainable parameters left. "
                f"Reduce 'freeze' or pass a list of specific layer indices."
            )

                # ---------------------------
        # AMP GradScaler
        # ---------------------------
        self.scaler =  torch.amp.GradScaler("cuda", enabled=self.amp)

        self.accumulate = max(round(self.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.weight_decay * self.batch_size * self.accumulate / self.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.optimizer_name,
            lr=self.lr0,
            momentum=self.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self._setup_scheduler()

        self.ema = ModelEMA(self.model)

        self.start_epoch = self.load_checkpoint()

        if self.start_epoch == 0:
            LOGGER.info("Starting fresh training.")
            # #self.initialize_weights()
            # self.model.train()  # Set model back to training(default) mode
            # unwrap_model(self.model).detect.bias_init()
            # #self.model = torch.compile(self.model, mode="default")
        else:
            LOGGER.info(f"Resuming from epoch {self.start_epoch}")

       
    def train(self, rank = 0, sampler=None):
        """Train the model"""
        self.train_time_start = time.time()
        self._setup_train()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.warmup_epochs * nb), 100) if self.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1

        self.optimizer.zero_grad(set_to_none=True)
        
        # ---------------------------
        # Training loop
        # ---------------------------
        for epoch in range(self.start_epoch, self.epochs):
            self.nan_found = False
            if sampler:
                sampler.set_epoch(epoch)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
                
            self._model_train()

            disable = (
                            dist.is_available()
                            and dist.is_initialized()
                            and dist.get_rank() != 0
                        )
                        
            pbar = tqdm(
                            enumerate(self.train_loader),
                            total=nb,
                            leave=False,
                            disable=disable,
                        )
            self.tloss = None
            for i, batch in pbar:
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])
        
                # ---------------------------
                # Forward + Loss (AMP safe)
                # ---------------------------
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    # ---------------------------
                    # Validate batch tensors
                    # ---------------------------
                    batch = self.preprocess_batch(batch)
                    
                    preds = self.model(batch['img'], mode='off_pipe')
                    loss, self.loss_items = self.criterion(preds, batch)
                    self.loss = loss.sum()
                    if dist.is_initialized():
                        self.loss *= dist.get_world_size()

                    
                    if not torch.isfinite(self.loss):
                        print(f"Invalid loss at batch {i}: {self.loss}")
                        self.nan_found = True
                        continue

                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)

                # ---------------------------
                # Backward
                # ---------------------------
                self.scaler.scale(self.loss).backward()
        
                if ni - last_opt_step >= self.accumulate:   
                    self.optimizer_step()
                    last_opt_step = ni
            
                # =============================
                # Display GLOBAL loss
                # =============================
                if self.tloss is not None:
                    tloss_disp = self.tloss.clone()
                
                    if dist.is_initialized():
                        dist.all_reduce(tloss_disp, op=dist.ReduceOp.SUM)
                        tloss_disp /= dist.get_world_size()
                
                    if RANK in {-1, 0}: 
                        mem = self._get_memory()
                        l = tloss_disp.tolist()
                        pbar.set_description(
                            f"{epoch+1}/{self.epochs} "
                            f"{mem:.2f}G "
                            f"box:{l[0]:.4f} cls:{l[1]:.4f} dfl:{l[2]:.4f} "
                            f" bs:{batch['cls_next'].shape[0]}"
                            f" img:{batch['img'].shape[2:]}"
                            f"lr:{self.optimizer.param_groups[0]['lr']:.6f}"
                        )
                self._clear_memory()

            # =====================================================
            # Recovery
            # =====================================================
            if self.nan_found:
    
                LOGGER.warning(
                    f"Recovering from checkpoint after "
                    f"NaN loss in epoch {epoch+1}"
                )
    
                if self._handle_nan_recovery(epoch):
                    continue
    
            self.nan_recovery_attempts = 0

            # Epoch summary (global)
            if self.tloss is not None:
                tloss_epoch = self.tloss.clone()
                if dist.is_initialized():
                    dist.all_reduce(tloss_epoch, op=dist.ReduceOp.SUM)
                    tloss_epoch /= dist.get_world_size()

            train_loss = {
                            "box_loss": tloss_epoch[0].cpu(),
                            "cls_loss": tloss_epoch[1].cpu(),
                            "dfl_loss": tloss_epoch[2].cpu(),
                        }
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
           
            self._clear_memory(None if self.device.type == "mps" else 0.5)  # prevent VRAM spike
            self.metrics, self.fitness = self.validate()
            results = {**train_loss, **self.metrics, "fitness":self.fitness}
            if RANK in {-1, 0}:   # True
                self.save_metrics(metrics= {**train_loss, **self.metrics, **self.lr},epoch=epoch)

                self.save_checkpoint(epoch + 1)
                print(
                        f"Epoch {epoch+1}/{self.epochs}: " +
                        " | ".join(f"{k}={float(v):.5f}" for k, v in results.items())
                     )
            # stop = self.stopper(epoch + 1, self.fitness)

            # if stop:
            #     LOGGER.info(
            #         f"Early stopping at epoch {epoch + 1}. "
            #         f"Best fitness = {self.stopper.best_fitness:.5f}"
            #     )
            #     break

            self._clear_memory()

        #self.plot_metrics()

            
    def preprocess_batch(self,  batch: dict, rank=0,is_distributed=True) -> dict:
        """Preprocess a batch of images by scaling and converting to float.
    
        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.
    
        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float().div_(255.0)
    
        for k in ("cls_curr","bboxes_curr","batch_idx_curr", "cls_next","bboxes_next","batch_idx_next"):
            batch[k] = batch[k].to(self.device, non_blocking=True)
        tensor = torch.LongTensor(2).cuda()
        imgs = batch["img"]
        if RANK in {-1, 0}: 
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())

        imgs = F.interpolate(
            imgs,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        
        batch["img"] = imgs
        return batch
        
    def validate(self):
        if self.ema and dist.is_initialized():
            # Sync EMA buffers from rank 0 to all ranks
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        
        metrics = DetectionValidator( model= self.ema.ema, 
                                   val_loader = self.val_loader, 
                                   loss= self.criterion, 
                                   device = self.device, 
                                   class_names = self.class_names,
                                   training=False,
                                   prefix='val').validate()
        if metrics is None:
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness
        

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()
    def read_results_csv(self):
        """Read results.csv into a dictionary using pandas."""
        import pandas as pd
    
        try:
            return pd.read_csv(self.csv).to_dict(orient="list")
        except Exception:
            return {}

    def save_checkpoint(self, epoch):
        import io
        """Overwrite model.pt after every successful epoch."""
        # Skip only genuinely non-finite fp32 EMA weights; fp16 overflow is handled on the saved copy below.
        # Skip only genuinely non-finite fp32 EMA weights; fp16 overflow is handled on the saved copy below.
        ema = unwrap_model(self.ema.ema)
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            LOGGER.warning(f"Skipping checkpoint save at epoch {self.epoch}: EMA contains NaN/Inf")
            return False
        ema = deepcopy(ema).half()
        # Clamp fp16 serialization overflow without mutating the live EMA.
        for v in ema.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v)
        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        ckpt = {
            "epoch": epoch,
            "best_fitness": self.best_fitness,
            "model": unwrap_model(self.model).state_dict(),
            "ema": ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_metrics": {**self.metrics},
            "train_results": self.read_results_csv(),
            "date": datetime.now().isoformat(),
        }
        torch.save(
            ckpt,
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save
        # Save checkpoints
        self.wdir.mkdir(parents=True, exist_ok=True)  # ensure weights directory exists
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
            
        print(f"Checkpoint saved -> {self.last}")
        
    
    def _handle_nan_recovery(self, epoch):
        """Detect and recover from NaN/Inf loss and fitness collapse by loading last checkpoint."""
        loss_epoch = self.loss.clone()
        if dist.is_initialized():
            dist.all_reduce(loss_epoch, op=dist.ReduceOp.SUM)
            loss_epoch /= dist.get_world_size()

        loss_nan = loss_epoch is not None and not loss_epoch.isfinite()
        # fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
        # fitness_collapse = self.best_fitness and self.best_fitness > 0 and self.fitness == 0
        corrupted = loss_nan #and RANK in {-1, 0} and (fitness_nan or fitness_collapse)
        reason = "Loss NaN/Inf" #if loss_nan #else "Fitness NaN/Inf" if fitness_nan else "Fitness collapse"
        # if RANK != -1:  # DDP: broadcast to all ranks
        #     broadcast_list = [corrupted if RANK == 0 else None]
        #     dist.broadcast_object_list(broadcast_list, 0)
        #     corrupted = broadcast_list[0]
        if not corrupted:
            return False
        if epoch == self.start_epoch:
            LOGGER.warning(f"{reason} detected but can not recover from last.pt...")
            return False  # Cannot recover on first epoch, let training continue
        if not self.ckpt_path.exists():
            raise RuntimeError(f"{reason} detected but no valid last.pt is available for recovery")
        self.nan_recovery_attempts += 1
        if self.nan_recovery_attempts > 3:
            raise RuntimeError(f"Training failed: NaN persisted for {self.nan_recovery_attempts} epochs")
        LOGGER.warning(f"{reason} detected (attempt {self.nan_recovery_attempts}/3), recovering from last.pt...")
        self._model_train()  # set model to train mode before loading checkpoint to avoid inference tensor errors
        self.start_epoch = self.load_checkpoint()
        # ema_state = ckpt["ema"].float().state_dict()
        # if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
        #     raise RuntimeError(f"Checkpoint {self.last} is corrupted with NaN/Inf weights")
        # unwrap_model(self.model).load_state_dict(ema_state)  # Load EMA weights into model
        # self._load_checkpoint_state(ckpt)  # Load optimizer/scaler/EMA/best_fitness
        # del ckpt, ema_state
        self.scheduler.last_epoch = epoch - 1
        return True

    def load_checkpoint(self):
        """Load checkpoint if available, otherwise start fresh."""
        
        if not self.last.exists():
            LOGGER.info("No checkpoint found. Starting fresh training.")
            return 0
    
        try:
            ckpt = torch.load(
                self.last,
                map_location="cpu",
                weights_only=False,
            )
    
            unwrap_model(self.model).load_state_dict(
                                                        ckpt["model"],
                                                        strict=True,
                                                    )
                                                        
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
    
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
    
            if "scaler" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler"])
    
            if self.ema and ckpt.get("ema") is not None:
                self.ema.ema.load_state_dict(ckpt["ema"])
    
            start_epoch = ckpt.get("epoch", 0)
            if RANK in {-1, 0}: 
    
                LOGGER.info(f"Resuming training from epoch {start_epoch}")
    
            return start_epoch
    
        except Exception as e:
            
            LOGGER.warning(
                f"Failed to load checkpoint ({e}). Starting fresh training."
            )
            return 0

    

    def initialize_weights(self):
        """Initialize model weights to random values."""
        for m in self.model.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.1
                
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU,nn.GELU}:
                m.inplace = True

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)



    def _clear_memory(self, threshold: float | None = None):
        """Clear accelerator memory by calling garbage collector and emptying cache."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()
    
    
    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema:
            self.ema.update(self.model)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.cos_lr:
            self.lf = one_cycle(1, self.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.lrf) + self.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def save_metrics(self, metrics, epoch):
        """Save training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        t = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)  # ensure parent directory exists
        s = "" if self.csv.exists() else ("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n"
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % (epoch + 1, t, *vals)).rstrip(",") + "\n")

    # def plot_metrics(self):
    #     """Plot metrics from a CSV file."""
    #     plot_results(file=self.csv, on_plot=self.on_plot(name=self.save_dir/'results.png' ))  # save results.png

    # def on_plot(self, name, data=None):
    #     """Register plots (e.g. to be consumed in callbacks)."""
    #     path = Path(name)
    #     self.plots[path] = {"data": data, "timestamp": time.time()}

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected based on the
                number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.
            iterations (float, optional): The number of iterations, which determines the optimizer if name is 'auto'.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [{}, {}, {}, {}]  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = 8  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        use_muon = name == "MuSGD"
        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if param.ndim >= 2 and use_muon:
                    g[3][fullname] = param  # muon params
                elif "bias" in fullname:  # bias (no decay)
                    g[2][fullname] = param
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1][fullname] = param
                else:  # weight (with decay)
                    g[0][fullname] = param
        if not use_muon:
            g = [x.values() for x in g[:3]]  # convert to list of params

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "MuSGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optim_args = dict(lr=lr, momentum=momentum)
        elif name == "SGD" or name == "MuSGD":
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        num_params = [len(g[0]), len(g[1]), len(g[2])]  # number of param groups
        g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
        g[0] = {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
        g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
        muon, sgd = (0.4355, 0.47908)
        if use_muon:
            num_params[0] = len(g[3])  # update number of params
            g[3] = {"params": g[3], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"}
            import re

            # higher lr for certain parameters in MuSGD when funetuning
            # proto.semseg is the checkpoint parameter name for YOLO26 semantic auxiliary heads.
            pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg|SemanticSegment")
            g_ = []  # new param groups
            for x in g:
                p = x.pop("params")
                p1 = [v for k, v in p.items() if pattern.search(k)]
                p2 = [v for k, v in p.items() if not pattern.search(k)]
                g_.extend([{"params": p1, **x, "lr": lr * 3}, {"params": p2, **x}])
            g = g_
        optimizer = getattr(optim, name, partial(MuSGD, muon=muon, sgd=sgd))(params=g)
        
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), {num_params[2]} bias(decay=0.0)"
        )
        return optimizer