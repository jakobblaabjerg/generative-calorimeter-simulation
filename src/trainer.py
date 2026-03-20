import torch
import os 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .losses import LOSSES

class Trainer:

    def __init__(self,
                 model,
                 optimizer,
                 run_dir,
                 epochs,
                 loss,
                 scheduler=None,
                 patience=None
                 ):

        self.model = model
        self.optimizer = optimizer
        self.run_dir = run_dir
        self.epochs = epochs
        self.loss = LOSSES[loss]
        self.patience = patience
        self.scheduler = scheduler

        print(self.patience, type(self.patience))

        self.writer = SummaryWriter(self.run_dir)           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def fit(self, train_loader, val_loader=None):

        best_val_loss = float("inf")
        early_stop_counter = 0

        for epoch in range(self.epochs):

            train_loss = self._training_loop(train_loader, epoch)

            self.writer.add_scalar("Loss/train", train_loss, epoch)

            for name, param in self.model.named_parameters():

                self.writer.add_histogram(f"{name}_weight", param.detach().cpu(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"{name}_grad", param.grad.detach().cpu(), epoch)

            if val_loader is not None:
                val_loss = self._validation_loop(val_loader, epoch)

                self.writer.add_scalar("Loss/val", val_loss, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    self._save_checkpoint(name="best")
                    print(f"Epoch {epoch+1}: New best model saved (val_loss={val_loss:.4f})")

                else:
                    early_stop_counter += 1

                if self.patience is not None and early_stop_counter >= self.patience: ################# not working
                    print("Early stopping triggered.")
                    break

            if self.scheduler is not None:
                self.scheduler.step()

        self._save_checkpoint(name="last")
        self.writer.close()


    def _training_loop(self, train_loader, epoch):

        self.model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

        for batch in loop:

            x, c = batch

            x=x.to(self.device)
            c=c.to(self.device)

            if torch.isnan(x).any():
                print("NaN in x")

            if torch.isnan(c).any():
                print("NaN in c")

            pi, mu, log_var = self.model(c)

            if torch.isnan(pi).any():
                print("NaN in pi")

            if torch.isnan(mu).any():
                print("NaN in mu")

            if torch.isnan(log_var).any():
                print("NaN in log_var")

            loss = self.loss(pi, mu, log_var, x)

            if torch.isnan(pi).any() or torch.isnan(mu).any() or torch.isnan(log_var).any():
                print(pi_last, mu_last, log_var_last, loss)
                print("xlast")
                print(x_last)
                print("clast")
                print(c_last)
                break


            pi_last = pi
            mu_last = mu
            log_var_last = log_var
            loss_last = loss
            x_last = x
            c_last = c

            # print(loss)

            self.optimizer.zero_grad()  # regires gradd
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

        train_loss = sum(train_losses) / len(train_losses)

        return train_loss


    def _validation_loop(self, val_loader, epoch):

        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:

                x, c = batch

                x=x.to(self.device)                        
                c=c.to(self.device)

                pi, mu, log_var = self.model(c)

                loss = self.loss(pi, mu, log_var, x)

                val_losses.append(loss.item())
                    
        val_loss = sum(val_losses) / len(val_losses)

        return val_loss


    def _save_checkpoint(self, name=""):

        file_path = os.path.join(self.run_dir, f"{name}_model.pt")
        torch.save(self.model.state_dict(), file_path)
