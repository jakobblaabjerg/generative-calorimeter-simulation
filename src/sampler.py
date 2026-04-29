import torch
from src.data_processing import concat_dict

class Sampler:

    def __init__(self,
                 model,
                 ):

        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device


    def sample(self, data_loader):

        data, meta = {}, {}

        for batch in data_loader:

            batch = batch.to(self.device) if torch.is_tensor(batch) else batch

            with torch.no_grad():

                data_b, meta_b = self.model.sample(batch)
                data = concat_dict(data, data_b)
                meta = concat_dict(meta, meta_b)
        
        return data, meta 
        