import torch
from src.data_processing import concat_dict, normalize_data, standardize_data, load_stats, compute_static_features, DataProcessor

class Sampler:

    def __init__(self,
                 model,
                 cfg_sampler,
                 ):
        
        self.model = model
        load_dir = cfg_sampler.data_loader.load_dir
        self.standardize_vars = cfg_sampler.data_loader.standardize_vars
        self.file_idx = 1 
        self.stats = load_stats(load_dir)
        self.processor = DataProcessor(save_dir=load_dir)
        self.num_files = cfg_sampler.num_files
        self.cfg_sampler = cfg_sampler
        self.device = next(model.parameters()).device

    def sample(self, data_loader):

        self.model.eval()
        data, meta = {}, {}

        for batch in data_loader:

            batch = batch.to(self.device) if torch.is_tensor(batch) else batch

            with torch.no_grad():

                data_b, meta_b = self.model.sample(batch)
                data = concat_dict(data, data_b)
                meta = concat_dict(meta, meta_b)

        self.data_sampled = data
        self.meta_sampled = meta


    def inverse_transform(self, cfg_filters):

        standardize_data(self.data_sampled, self.stats, self.standardize_vars, inverse=True)
        standardize_data(self.meta_sampled, self.stats, self.standardize_vars, inverse=True)

        normalize_data(self.data_sampled, self.meta_sampled, cfg_filters, inverse=True)
        compute_static_features(self.data_sampled, self.meta_sampled, inverse=True)



    def save_data(self):

        self.processor.file_idx = self.file_idx
        self.processor.data_sampled = self.data_sampled
        self.processor.meta_sampled = self.meta_sampled
        self.processor.save_data(stage="sampled")
        self.file_idx += 1