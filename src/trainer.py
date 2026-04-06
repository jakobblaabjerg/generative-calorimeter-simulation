import torch
import os 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

        self.writer = SummaryWriter(self.run_dir)           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    
    def fit(self, train_loader, val_loader=None):

        best_val_loss = float("inf")
        early_stop_counter = 0

        for epoch in range(self.epochs):

            train_loss_total, train_loss_components = self._training_loop(train_loader, epoch)

            self.writer.add_scalar("Loss/train/total", train_loss_total, epoch)
            for i, l in enumerate(train_loss_components):
                self.writer.add_scalar(f"Loss/train/(component_{i+1}", l, epoch)

            for name, param in self.model.named_parameters():

                self.writer.add_histogram(f"{name}_weight", param.detach().cpu(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"{name}_grad", param.grad.detach().cpu(), epoch)

            if val_loader is not None:

                val_loss_total, val_loss_components = self._validation_loop(val_loader, epoch)

                self.writer.add_scalar("Loss/val/total", val_loss_total, epoch)
                for i, l in enumerate(val_loss_components):
                    self.writer.add_scalar(f"Loss/val/component_{i+1}", l, epoch)

                if val_loss_total < best_val_loss:
                    best_val_loss = val_loss_total
                    early_stop_counter = 0
                    self._save_checkpoint(epoch, which="best")
                    print(f"Epoch {epoch+1}: New best model saved (val_loss={val_loss_total:.4f})")

                else:
                    early_stop_counter += 1

                if self.patience is not None and early_stop_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            if self.scheduler is not None:
                self.scheduler.step()

        self._save_checkpoint(epoch, which="last")
        self.writer.close()


    def _training_loop(self, train_loader, epoch):

        self.model.train()       
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

        for batch_idx, batch in enumerate(loop):

            batch = tuple(t.to(self.device) if torch.is_tensor(t) else t for t in batch)
            losses = self.model(*batch)

            if not isinstance(losses, (tuple, list)):
                losses = (losses,)
            
            if batch_idx == 0:
                train_losses = [[] for _ in range(len(losses))]

            for i, l in enumerate(losses):
                train_losses[i].append(l.item())

            l_total = sum(losses)

            self.optimizer.zero_grad() 
            l_total.backward()
            self.optimizer.step()

            loop.set_postfix(loss=l_total.item())

        train_loss_components = [sum(l)/len(l) for l in train_losses]
        train_loss_total = sum(train_loss_components)

        return train_loss_total, train_loss_components


    def _validation_loop(self, val_loader, epoch):

        self.model.eval()
        loop = tqdm(val_loader, leave=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(loop):

                batch = tuple(t.to(self.device) if torch.is_tensor(t) else t for t in batch)
                losses = self.model(*batch)

                if not isinstance(losses, (tuple, list)):
                    losses = (losses,)

                if batch_idx == 0:
                    val_losses = [[] for _ in range(len(losses))]

                for i, l in enumerate(losses):
                    val_losses[i].append(l.item())

                l_total = sum(losses)
                
                loop.set_postfix(loss=l_total.item())   

        val_loss_components = [sum(l)/len(l) for l in val_losses]
        val_loss_total = sum(val_loss_components)

        return val_loss_total, val_loss_components


    def _save_checkpoint(self, epoch, which=""):

        file_path = os.path.join(self.run_dir, f"{which}_model.pt")

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            }, file_path
            )



