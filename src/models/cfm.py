from .blocks import MLP
from .base import BaseModel

from .mdn import MixtureDensityNetworkV1, MixtureDensityNetworkV2
from . import encoders
from .registry import register_model, ENCODER_REGISTRY, MODEL_REGISTRY

from src.data.datasets import get_feature_names 
from src.config import load_config
from src.calosim import CaloSimDataset

import torch
import numpy as np

@register_model("cfm")
class ConditionalFlowMatching(BaseModel):

    def __init__(self, cfg):

        super().__init__()

        self.num_voxels = getattr(cfg, "num_voxels", None) # only for sampling! 

        # auxillary model used for inference     
        self.aux_model = None
        self.aux_model_dir = cfg.aux_model.model_dir
        self.aux_model_name = cfg.aux_model.name

        # number integration steps 
        self.num_steps = cfg.num_steps

        # variable names for z and c
        _, self.z_vars, self.c_vars = get_feature_names(cfg.input_vars, cfg.transforms)

        # input dimensions 
        self.point_dim = len(self.z_vars)
        self.cond_dim = len(self.c_vars)
        
        # neural nets        
        self.cfg_encoder = cfg.encoder
        self.encoder = self.create_encoder(self.cfg_encoder) if self.cfg_encoder is not None else None
  
        self.track_history = cfg.track_history

        self.mlp = MLP(
            hidden_layers=cfg.mlp.hidden_layers, 
            layer_norm=cfg.mlp.layer_norm, 
            input_size=self.mlp_input_size, 
            output_size=self.point_dim,
            activation=cfg.mlp.activation,
            )


    def create_encoder(self, cfg):

        self.use_cond = cfg.use_cond # t, c
        self.encoder_name = cfg.name

        input_size = self.point_dim
        if self.use_cond:
            input_size += self.cond_dim + 1

        encoder_cls = ENCODER_REGISTRY[self.encoder_name]            
        cfg = {key: value for key, value in vars(cfg).items() if key not in ["name", "use_cond"]}
        encoder = encoder_cls(input_size=input_size, **cfg)

        return encoder


    @property
    def mlp_input_size(self):
       
        if self.encoder is None:
            return self.point_dim + self.cond_dim + 1

        if self.encoder_name in ["sequence"]:
            return self.encoder.output_size

        if self.encoder_name in ["deepsets", "pointnet"]:
            return self.point_dim + self.cond_dim + 1 + self.encoder.output_size

        raise ValueError(f"Unknown encoder: {self.encoder_name}")

    @staticmethod
    def z_t(z_0, z_1, t):
        return t * z_1 + (1-t) * z_0


    @staticmethod
    def v_t(z_0, z_1):
        return z_1 - z_0


    def v_theta(self, z_t, t, c, num_points):
        
        if self.encoder is None:
            inputs = torch.cat([z_t, t, c], dim=-1)
            loss_reg = torch.tensor(0.0, device=z_t.device)

        else:
            inputs, loss_reg = self.encoder(z_t, t, c, num_points)

        return self.mlp(inputs), loss_reg


    def forward(self, x, z, c, num_points):        

        device = z.device
        batch_size= c.size(0)
        
        c_repeated = torch.repeat_interleave(c, num_points, dim=0)

        # sample the time step per batch element
        t = torch.rand(batch_size, device=device)
        t = torch.repeat_interleave(t.unsqueeze(-1), num_points, dim=0)

        # sample z_0 from p_0
        z_0 = torch.randn_like(z)

        z_t = self.z_t(z_0, z, t)
        v_t = self.v_t(z_0, z) 
        v_theta, loss_reg = self.v_theta(z_t, t, c_repeated, num_points)

        loss = self.loss(v_theta, v_t, num_points) + loss_reg
        
        return loss


    def loss(self, v_theta, v_t, num_points):
        
        loss = ((v_theta - v_t)**2).sum(dim=-1)  # total loss per point 
        loss = torch.segment_reduce(loss, reduce="mean", lengths=num_points)
        
        return torch.mean(loss)


    def _load_model_aux(self, device):

        # load config file of auxillary model         
        cfg_aux_model = load_config(f"{self.aux_model_dir}/config.yaml")

        # load auxillary model
        self.aux_model = MODEL_REGISTRY[self.aux_model_name](cfg_aux_model.model) # use factory instead
        self.aux_model.load_checkpoint(self.aux_model_dir)
        self.aux_model.to(device)

    def sample_num_points(self, c):

        device = c.device

        if self.num_voxels is not None:

            return torch.full(
                (c.shape[0],),
                self.num_voxels,
                dtype=torch.long,
                device=device,
            )
        
        # load aux model if it does not exist
        if self.aux_model is None:
            self._load_model_aux(device)

        # sample from auxilary model
        return self.aux_model.sample_num_points(c)


    def sample_noise(self, num_points):

        device = num_points.device
        total_points = num_points.sum().item()
        shape = (total_points, self.point_dim)
        z_t = torch.distributions.Normal(0,1).sample(shape).to(device)

        return z_t


    def solve_ode(self, z_t, c, num_points):

        snapshot_times = [0.0, 0.5, 1.0]
        steps = {round(t * (self.num_steps - 1)) for t in snapshot_times}
        states = []

        device = num_points.device
        total_points = num_points.sum().item()
        c_repeated = torch.repeat_interleave(c, num_points, dim=0) 

        # step size   
        delta_t = torch.full((total_points, 1), 1/self.num_steps, device=device)

        # integrate forward Euler
        for i in range(self.num_steps):            
            
            t = (i+1)*delta_t
            v_theta, _ = self.v_theta(z_t, t, c_repeated, num_points)

            if self.track_history and i in steps:
                  states.append((z_t.detach().clone(), v_theta.detach()))
                              
            z_t += v_theta * delta_t

        return z_t, states


    def to_dataset(self, z, c, num_points, states):
        
        data, meta = {}, {}

        z = z.cpu().numpy()      
        c = c.cpu().numpy()
        states = [(z.cpu().numpy(), v.cpu().numpy()) for z, v in states]
        num_points = num_points.cpu().numpy().astype(int)  

        for j, var in enumerate(self.z_vars):
            data[var] = z[:, j]

            if self.track_history:
                data[f"{var}_hist"] = np.stack([s[0][:, j] for s in states], axis=1)
                data[f"v_{var}_hist"] = np.stack([s[1][:, j] for s in states], axis=1)

        for j, var in enumerate(self.c_vars):
            meta[var] = c[:, j]    

        meta["idx"] = np.arange(len(num_points))
        data["idx"] = np.repeat(meta["idx"], num_points)

        return CaloSimDataset(data=data, meta=meta)

    def sample(self, c):
         
        num_points = self.sample_num_points(c)
        z_0 = self.sample_noise(num_points)
        z, velocities = self.solve_ode(z_0, c, num_points)

        return self.to_dataset(z, c, num_points, velocities)