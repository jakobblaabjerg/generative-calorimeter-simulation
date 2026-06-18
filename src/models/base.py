import torch
import os 


class BaseModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return next(self.parameters()).device
     
    def load_checkpoint(self, load_dir, which="best"):

        file_path = os.path.join(
            load_dir, 
            f"{which}_model.pt"
        )

        checkpoint = torch.load(
            file_path, 
            map_location=self.device
        )        

        self.load_state_dict(
            checkpoint["model_state_dict"]
        )

    def save_checkpoint(self, save_dir, optimizer, epoch, which):

        file_path = os.path.join(
            save_dir,
            f"{which}_model.pt"
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            },
            file_path
        )



    # i am ot sure this is workig since mlp does ot have 

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)