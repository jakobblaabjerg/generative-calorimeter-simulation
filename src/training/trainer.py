from .loops import run_epoch

from torch.utils.tensorboard import SummaryWriter


import torch

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
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, loss):

        improved = loss < self.best_loss

        if improved:
            self.best_loss = loss
            self.counter = 0

        else:
            self.counter += 1

        return improved

    @property
    def should_stop(self):
        return (
            self.patience is not None
            and self.counter >= self.patience
        )          

  
class Trainer:

    def __init__(self,
                 model,
                 optimizer,
                 run_dir,
                 epochs,
                 scheduler=None,
                 patience=None
                 ):

        self.model = model     
        self.optimizer = optimizer
        self.run_dir = run_dir
        self.epochs = epochs
        self.scheduler = scheduler

        model_summary(self.model)

        self.writer = SummaryWriter(self.run_dir)           
        self.early_stopping = EarlyStopping(patience)


    def validate(self, val_loader, debug=False):

        if debug:
            # remember current state
            state_cpu = torch.random.get_rng_state()
            state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            
            set_seed()
            loss_val = run_epoch(self.model, val_loader) 

            # restore state
            torch.random.set_rng_state(state_cpu)
            if state_cuda is not None:
                torch.cuda.set_rng_state_all(state_cuda)

        else:
            loss_val = run_epoch(self.model, val_loader) 

        return loss_val


    def fit(self, train_loader, val_loader=None, debug=False):

        for epoch in range(self.epochs):
            
            desc = f"Epoch {epoch+1}/{self.epochs}"
            loss_train = run_epoch(self.model, train_loader, self.optimizer, desc)
            self.log_metrics(loss_train, epoch, tag="train", log_histograms=False)


            if val_loader is not None:

                loss_val = self.validate(val_loader, debug)
                self.log_metrics(loss_val, epoch, tag="val")                
                
                if self.early_stopping.step(sum(loss_val)):
                    self.model.save_checkpoint(self.run_dir, self.optimizer, epoch, which="best")
                    print(f"Epoch {epoch+1}: New best model saved (val_loss={sum(loss_val):.4f})")

                if self.early_stopping.should_stop:
                    print("Early stopping triggered.")
                    break

            if self.scheduler is not None:
                self.scheduler.step()
        
        self.model.save_checkpoint(self.run_dir, self.optimizer, epoch, which="last")
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

    

def run_train(cfg, debug=False):

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
    optimizer = create_optimizer(model, cfg.optimizer)


    print("Starting training")
    trainer = Trainer(model, optimizer, run_dir, **vars(cfg.trainer))
    trainer.fit(train_loader, val_loader, debug=debug)