from .loops import run_epoch

from torch.utils.tensorboard import SummaryWriter


import torch
import math
import shutil
import os

from src.utils import set_seed
from src.logger import Logger
from src.config import save_config
from src.optimizers import create_optimizer
from src.data.datasets import create_loader
from src.models.registry import MODEL_REGISTRY     # does this work !? 
from src.models import cfm, mdn
from src.reporting import model_summary


class EarlyStopping:

    def __init__(self, patience):

        self.patience = patience
        self.counter = 0

    def step(self, improved):

        if improved:
            self.counter = 0
        else:
            self.counter += 1

    @property
    def should_stop(self):
        return (
            self.patience is not None
            and self.counter >= self.patience
        )       


def create_scheduler(optimizer, total_steps, config):

    warmup_steps = int(config.warmup_fraction * total_steps)

    if warmup_steps == 0:

        if config.schedule == "constant":
            return torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=total_steps,
            )

        elif config.schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=config.min_lr,
            )

        else:
            raise ValueError(
                f"Unknown learning rate schedule: {config.schedule}"
            )

    # warmup
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    remaining_steps = total_steps - warmup_steps

    # after warmup
    if config.schedule == "constant":

        after_warmup = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=remaining_steps,
        )

    elif config.schedule == "cosine":

        after_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_steps,
            eta_min=config.min_lr,
        )

    else:
        raise ValueError(
            f"Unknown learning rate schedule: {config.schedule}"
        )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, after_warmup],
        milestones=[warmup_steps],
    )


class Trainer:

    def __init__(self,
                 model,
                 run_dir,
                 epochs,
                 optimizer,
                 patience=None,
                 scheduler=None,
                 ):

        self.model = model 
        self.run_dir = run_dir
        self.epochs = epochs
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        self.optimizer = create_optimizer(model, self.optimizer_config)
        self.writer = SummaryWriter(self.run_dir)    
        self.early_stopping = EarlyStopping(patience)

        model_summary(self.model)

    def validate(self, val_loader, seed=None):

        if seed is not None:
            # remember current state
            state_cpu = torch.random.get_rng_state()
            state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            
            set_seed(seed) 
            loss_val = run_epoch(self.model, val_loader) 

            # restore state
            torch.random.set_rng_state(state_cpu)
            if state_cuda is not None:
                torch.cuda.set_rng_state_all(state_cuda)

        else:
            loss_val = run_epoch(self.model, val_loader) 

        return loss_val


    def fit(self, train_loader, val_loader=None, seed=None):
        
        best_val_loss = float("inf")

        if self.scheduler_config is not None:
            total_steps = self.epochs * len(train_loader)
            self.scheduler = create_scheduler(self.optimizer, total_steps, self.scheduler_config)
        else:
            self.scheduler = None

        for epoch in range(self.epochs):

            for i, group in enumerate(self.optimizer.param_groups):
                print(f"Group {i}: lr = {group['lr']}")


            desc = f"Epoch {epoch+1}/{self.epochs}"
            loss_train = run_epoch(self.model, train_loader, self.optimizer, self.scheduler, desc)
            self.log_metrics(loss_train, epoch, tag="train", log_histograms=False)

            # check for nan
            if math.isnan(sum(loss_train)):
                raise RuntimeError(f"Loss is NaN")

            if val_loader is not None:

                # compute and log validation loss
                loss_val = self.validate(val_loader, seed)
                self.log_metrics(loss_val, epoch, tag="val")                

                # monitor validation loss
                improved = sum(loss_val) < best_val_loss

                # save new checkpoint if improved
                if improved:
                    best_val_loss = sum(loss_val)
                    self.model.save_checkpoint(self.run_dir, self.optimizer, self.scheduler, epoch, which="best")
                    print(f"Epoch {epoch+1}: New best model saved (val_loss={sum(loss_val):.4f})")

                # early stopping  
                self.early_stopping.step(improved) 

                if self.early_stopping.should_stop: 
                    print("Early stopping triggered.")
                    break

        # save last checkpoint 
        self.model.save_checkpoint(self.run_dir, self.optimizer, self.scheduler, epoch, which="last")
        self.writer.close()


    def log_metrics(self, loss, epoch, tag, log_histograms=False):

        self.writer.add_scalar(f"Loss/{tag}/total", sum(loss), epoch)        
        
        for i, l in enumerate(loss):
            self.writer.add_scalar(f"Loss/{tag}/component_{i+1}", l, epoch)

        if log_histograms:
            for name, param in self.model.named_parameters():    
                self.writer.add_histogram(f"{name}_weight", param.detach().cpu(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"{name}_grad", param.grad.detach().cpu(), epoch)

    

def run_train(cfg, seed=None):

    print("Loading configuration")
    logger = Logger(**vars(cfg.logger))
    run_dir = logger.get_run_dir()
    cfg.run_dir = run_dir
    save_config(cfg, run_dir)

    print("Setting up loaders")
    train_loader = create_loader(split="train", **vars(cfg.data_loader))
    val_loader = create_loader(split="val", **vars(cfg.data_loader))

    print("Initializing model")
    model = MODEL_REGISTRY[cfg.name](cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Starting training")
    trainer = Trainer(model, run_dir, **vars(cfg.trainer))
    trainer.fit(train_loader, val_loader, seed)

    # save stats
    shutil.copy2(
        os.path.join(cfg.data_loader.load_dir, "stats.json"),
        os.path.join(cfg.run_dir, "stats.json"),
    )

# free, total = torch.cuda.mem_get_info()
# print(f"Free:  {free / 1024**3:.2f} GB")
# print(f"Total: {total / 1024**3:.2f} GB")