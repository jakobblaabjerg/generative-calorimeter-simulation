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
        



    # self.standardize_vars = standardize_vars
    # self.stats = stats 

    # self.file_idx = 1 


    # # is this is everythinkg
    # self.num_files = cfg_sampling.num_files
    # self.device = next(model.parameters()).device


    # def inverse_transform(self, cfg_filters):

    #     standardize_data(self.data_sampled, self.stats, self.standardize_vars, inverse=True)
    #     standardize_data(self.meta_sampled, self.stats, self.standardize_vars, inverse=True)

    #     normalize_data(self.data_sampled, self.meta_sampled, cfg_filters, inverse=True)
    #     compute_static_features(self.data_sampled, self.meta_sampled, inverse=True)



    # def save_data(self):

    #     self.processor.file_idx = self.file_idx
    #     self.processor.data_sampled = self.data_sampled
    #     self.processor.meta_sampled = self.meta_sampled
    #     self.processor.save_data(stage="sampled")
    #     self.file_idx += 1