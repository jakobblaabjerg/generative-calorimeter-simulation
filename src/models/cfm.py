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
from src.solvers import SOLVERS

@register_model("cfm")
class ConditionalFlowMatching(BaseModel):

    def __init__(self, cfg):

        super().__init__()


        # ONLY RELEVANT FOR SAMPLING. 
        # This decides wheter we are using point or voxel. 
        self.num_voxels = getattr(cfg, "num_voxels", None) # voxel-based.
        self.use_aux_model = getattr(cfg, "aux_model", None) is not None # point-based.
        
        if self.num_voxels is not None and self.use_aux_model:
            raise ValueError(
                "Both num_voxels and aux_model are set."
            )

        # get aux model
        if self.use_aux_model:
            self.aux_model = None
            self.aux_model_dir = cfg.aux_model.model_dir
            self.aux_model_name = cfg.aux_model.name

        # solver
        self.solver = cfg.solver

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
    def X_t(X_0, X_1, t):
        return t * X_1 + (1-t) * X_0

    @staticmethod
    def v_t(X_0, X_1):
        return X_1 - X_0

    def v_model(self, X_t, t, context, num_points):
        
        if self.encoder is None:
            inputs = torch.cat([X_t, t, context], dim=-1)
            loss_reg = torch.tensor(0.0, device=X_t.device)

        else:
            inputs, loss_reg = self.encoder(X_t, t, context, num_points)

        return self.mlp(inputs), loss_reg


    def forward(self, X_raw, X_1, context, num_points):        
    
        device = X_1.device
        batch_size= context.size(0)
        
        context_rep = torch.repeat_interleave(context, num_points, dim=0)

        # sample the time step per batch element
        t = torch.rand(batch_size, device=device)
        t = torch.repeat_interleave(t.unsqueeze(-1), num_points, dim=0)

        # sample X_0 from p_0
        X_0 = torch.randn_like(X_1)

        X_t = self.X_t(X_0, X_1, t)
        v_t = self.v_t(X_0, X_1) 
        v_model, loss_reg = self.v_model(X_t, t, context_rep, num_points)

        loss = self.loss(v_model, v_t, num_points) + loss_reg
        
        return loss


    def loss(self, v_model, v_t, num_points):
        
        loss = ((v_model - v_t)**2).sum(dim=-1)  # total loss per point 
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

        if self.use_aux_model:

            # load aux model if it does not exist
            if self.aux_model is None:
                self._load_model_aux(device)

            return self.aux_model.sample_num_points(c)


        elif self.num_voxels is not None:

            return torch.full(
                (c.shape[0],),
                self.num_voxels,
                dtype=torch.long,
                device=device,
            )
        
        else:
            raise ValueError(
                "Neither num_voxels nor aux_model are specified"
            )

    def sample_noise(self, num_points):

        device = num_points.device
        total_points = num_points.sum().item()
        shape = (total_points, self.point_dim)
        noise = torch.distributions.Normal(0,1).sample(shape).to(device)

        return noise




 
        

    def to_dataset(self, X_1, context, num_points, history):
        
        data, meta = {}, {}

        X_1 = X_1.cpu().numpy()      
        context = context.cpu().numpy()

        if history:

            history = [(
                h["X_t"].cpu().numpy(),
                h["v_t"].cpu().numpy(),
                )
                for h in history
            ]

        num_points = num_points.cpu().numpy().astype(int)  

        for j, var in enumerate(self.z_vars):
            data[var] = X_1[:, j]

            if self.track_history and history:
                data[f"{var}_his"] = np.stack([h[0][:, j] for h in history], axis=1)
                data[f"v_{var}_his"] = np.stack([h[1][:, j] for h in history], axis=1)


        for j, var in enumerate(self.c_vars):
            meta[var] = context[:, j]    

        meta["idx"] = np.arange(len(num_points))
        data["idx"] = np.repeat(meta["idx"], num_points)

        return CaloSimDataset(data=data, meta=meta)



    def sample(self, context):

         # sample number of points per point cloud
        num_points = self.sample_num_points(context) 

        # sample gaussian noise
        noise = self.sample_noise(num_points) 

        # solve the ode
        X_1, history = self.solve_ode(noise, context, num_points) 

        # convert to dataset
        dataset = self.to_dataset(X_1, context, num_points, history)

        return dataset


    def solve_ode(self, noise, context, num_points):

        try:
            solver = SOLVERS[self.solver](self.num_steps, self.track_history)
        except KeyError:
            raise ValueError(f"Unknown solver: {self.solver!r}")

        # repeat context vector 
        context_rep = torch.repeat_interleave(context, num_points, dim=0) 

        # create function of X and t. Context and num_points are fixed. 
        def velocity_func(X, t):
            v, _ = self.v_model(X, t, context_rep, num_points)
            return v
        
        return solver.solve(func=velocity_func, noise=noise)

