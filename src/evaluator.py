from src.datasets import get_data_loader
from src.models import load_checkpoint

import torch
from tqdm import tqdm

class Evaluator:

    def __init__(self, model_cls, cfg):

        self.cfg = cfg
        self.model = model_cls(**vars(self.cfg.model))

        self.device = torch.device(self.cfg.sampler.device)
        self.model.to(self.device)
        load_checkpoint(self.cfg.run_dir, self.model, self.device)
        self.model.eval()


    def evaluate(self, split):

        data_loader = get_data_loader(split=split, **vars(self.cfg.data_loader))

        loop = tqdm(data_loader, leave=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(loop):

                batch = tuple(t.to(self.device) if torch.is_tensor(t) else t for t in batch)
                loss = self.model(*batch)

                if not isinstance(loss, (tuple, list)):
                    loss = (loss,)

                if batch_idx == 0:
                    loss_list = [[] for _ in range(len(loss))]

                for i, l in enumerate(loss):
                    loss_list[i].append(l.item())

                loss_total = sum(loss)
                loop.set_postfix(loss=loss_total.item())

        loss_components = [sum(l)/len(l) for l in loss_list]        
        return sum(loss_components)


    def version(self):  
        return self.cfg.run_dir.split("_")[-1]


    def num_params(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())