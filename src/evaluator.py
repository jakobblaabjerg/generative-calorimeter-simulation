from src.datasets import get_data_loader
from src.models import load_checkpoint

import torch
from tqdm import tqdm


# combine into the 

class Evaluator:

    def __init__(self, model_cls, cfg):

        self.cfg = cfg
        self.model = model_cls(self.cfg.model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # change this!
        self.model.to(self.device)
        
        load_checkpoint(self.cfg.run_dir, self.model, self.device)
        self.model.eval()


    def evaluate(self, split):

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        data_loader = get_data_loader(split=split, **vars(self.cfg.data_loader))

        loop = tqdm(data_loader, leave=False)

        total_losses = None
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loop):

                num_batches += 1 

                batch = tuple(t.to(self.device) if torch.is_tensor(t) else t for t in batch)
                loss = self.model(*batch)

                if not isinstance(loss, (tuple, list)):
                    loss = (loss,)

                if total_losses is None:
                    total_losses = [0.0 for _ in range(len(loss))]

                for i, l in enumerate(loss):
                    total_losses[i] += l.item()

                loop.set_postfix(loss=sum(l.item() for l in loss))

        loss_components = [l/num_batches for l in total_losses]        
        return sum(loss_components)


    def version(self):  
        return self.cfg.run_dir.split("_")[-1]


    def num_params(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())