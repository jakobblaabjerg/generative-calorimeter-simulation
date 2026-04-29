import torch
import os 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from contextlib import nullcontext

from src.models import MODEL_REGISTRY, load_checkpoint
from src.datasets import get_data_loader


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
        self.patience = patience
        self.scheduler = scheduler

        print_model_summary(model=self.model)
        self.writer = SummaryWriter(self.run_dir)           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def log_metrics(self, loss_components, epoch, tag, log_histograms=False):

        self.writer.add_scalar(f"Loss/{tag}/total", sum(loss_components), epoch)        
        
        for i, l in enumerate(loss_components):
            self.writer.add_scalar(f"Loss/{tag}/component_{i+1}", l, epoch)

        if log_histograms:
            for name, param in self.model.named_parameters():    
                self.writer.add_histogram(f"{name}_weight", param.detach().cpu(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"{name}_grad", param.grad.detach().cpu(), epoch)

    
    def update_early_stopping(self, loss, best_loss, counter, epoch):
   
        if loss < best_loss:
            best_loss = loss
            counter = 0
            self._save_checkpoint(epoch, which="best")
            print(f"Epoch {epoch+1}: New best model saved (val_loss={loss:.4f})")
        
        else:
            counter += 1

        return best_loss, counter


    def fit(self, train_loader, val_loader=None, debug=False):

        loss_best = float("inf")
        counter = 0 # early stopping! 

        for epoch in range(self.epochs):
            desc = f"Epoch {epoch+1}/{self.epochs}"
            loss_train, comps_train = run_epoch(self.model, train_loader, self.device, self.optimizer, desc)
            self.log_metrics(comps_train, epoch, tag="train", log_histograms=False)

            if val_loader is not None:

                # for debugging purpose 
                if debug:
                    state_cpu = torch.random.get_rng_state()
                    state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    set_seed()

                    loss_val, comps_val = run_epoch(self.model, val_loader, self.device) # single-sample estimate

                    # restore state
                    torch.random.set_rng_state(state_cpu)
                    if state_cuda is not None:
                        torch.cuda.set_rng_state_all(state_cuda)

                else:
                    loss_val, comps_val = run_epoch(self.model, val_loader, self.device) # single-sample estimate

                self.log_metrics(comps_val, epoch, tag="val")                
                loss_best, counter = self.update_early_stopping(loss_val, loss_best, counter, epoch)

                if self.patience is not None and counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            if self.scheduler is not None:
                self.scheduler.step()

        self._save_checkpoint(epoch, which="last")
        self.writer.close()


    def _save_checkpoint(self, epoch, which=""):

        file_path = os.path.join(self.run_dir, f"{which}_model.pt")

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            }, file_path
            )


def accumulate_loss(loss_components, loss_running):

    if loss_running is None:
        loss_running = [0.0 for l in loss_components]

    for i, l in enumerate(loss_components):
        loss_running[i] += l

    return loss_running


def run_epoch(model, data_loader, device, optimizer=None, desc="", postfix_key="loss"):

    iterator = tqdm(data_loader, desc=desc, leave=False)
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss_running = None

    num_batches = 0

    grad_context = torch.no_grad() if not is_train else nullcontext()

    with grad_context:
        for batch_idx, batch in enumerate(iterator):
            loss_total, loss_components = run_step(model, batch, device, optimizer)
            loss_running = accumulate_loss(loss_components, loss_running)
            iterator.set_postfix({postfix_key: loss_total})
            num_batches += 1

    # normalize loss
    loss_running = [l/num_batches for l in loss_running] # this is a biased estimate
    
    return sum(loss_running), loss_running


def run_step(model, batch, device, optimizer=None):

    batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
    loss_batch = model(*batch)

    loss_components = loss_batch if isinstance(loss_batch, (tuple, list)) else (loss_batch,)
    loss_total = sum(loss_components)

    if optimizer is not None:
        optimizer.zero_grad() 
        loss_total.backward()
        optimizer.step()

    # detach from computational graph
    loss_total = loss_total.detach().item()
    loss_components = [l.detach().item() for l in loss_components]

    return loss_total, loss_components


def set_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_eval(cfg, split, num_samples, seed=None):

    run_dir = cfg.run_dir
    model_name = cfg.name 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODEL_REGISTRY[model_name](cfg.model)
    load_checkpoint(run_dir, model, device)

    data_loader = get_data_loader(split=split, **vars(cfg.data_loader))

    losses = []

    for i in range(num_samples):

        if seed is not None:
            set_seed(seed=seed+i)

        loss, _ = run_epoch(model, data_loader, device)
        losses.append(loss)
    
    loss_mean = sum(losses) / len(losses)
    loss_std = (sum((x - loss_mean)**2 for x in losses)/len(losses))**0.5
    num_params = get_num_params(model)

    metrics = {
        "loss_mean": loss_mean,
        "loss_std": loss_std,
        "num_samples": num_samples,
        "num_params": num_params,
    } 

    return metrics


def get_num_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model):
    
    total_params = 0
    trainable_params = 0

    print("\nModel Summary")
    print("-"*70)

    for name, module in model.named_children():        
        n_params = get_num_params(module)
        n_trainable = get_num_params(module, trainable_only=True)
        total_params += n_params
        trainable_params += n_trainable
        print(f"{name:<20} | params: {n_params:>10} | trainable: {n_trainable:>10}")
    
    print("-"*70)
    print(f"{'Total':<20} | params: {total_params:>10} | trainable: {trainable_params:>10}")
    print()



    




    