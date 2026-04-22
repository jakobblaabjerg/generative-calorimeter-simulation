import torch, math
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 
import os

from src.config import load_config
from src.datasets import setup_var_names

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

LOG_2PI = math.log(2 * math.pi)

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}


def make_jac_func(transform: str):
    if transform == "":
        return None
    elif transform == "sqrt":
        coef = -0.5
    elif transform == "log":
        coef = -1.0
    else:
        raise ValueError(f"Unknown transform {transform}")
    return lambda x: coef * torch.log(x)


class MixtureDensityNetwork(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.k = cfg.k
        self.alpha = cfg.alpha 
        self.add_jacobian = cfg.add_jacobian
       
        if self.add_jacobian:
            self.z_hat_jac_func = make_jac_func(cfg.transforms.z_hat)
            self.e_jac_func = make_jac_func(cfg.transforms.e)
       
        _, self.z_vars, self.c_vars = setup_var_names(cfg.transforms, cfg.spherical)
        self.point_dim = len(self.z_vars)

        self.mlp = MLP(
            hidden_layers=cfg.hidden_layers, 
            layer_norm=cfg.layer_norm, 
            input_size=cfg.cond_dim, 
            output_size=1+self.k*(2*self.point_dim+1), 
            activation=cfg.activation
            )


    def split(self, out):

        batch_size = out.size(0)

        idx = 0

        rate = out[:, idx:idx+1]
        idx += 1

        pi = out[:, idx:idx+self.k]
        idx += self.k

        means = out[:, idx:idx+self.k*self.point_dim]
        means = means.view(batch_size, self.k, self.point_dim)
        idx += self.k*self.point_dim

        log_vars = out[:, idx:idx+self.k*self.point_dim]
        log_vars = log_vars.view(batch_size, self.k, self.point_dim)

        return rate, pi, means, log_vars
  

    def loss_point(self, pi, means, log_vars, z, x):

        log_jacobian = torch.zeros_like(x)        

        if self.add_jacobian and self.e_jac_func is not None:
            log_jacobian[:, -1] = self.e_jac_func(x[:, -1])
        if self.add_jacobian and self.z_hat_jac_func is not None:
            log_jacobian[:, -2] = self.z_hat_jac_func(x[:, -2])

        log_jacobian = log_jacobian.sum(dim=1)

        _, k, _ = means.shape
        z = z.unsqueeze(1).expand(-1, k, -1)  # (batch, k, point_dim)
        
        # gaussian log probability
        inv_var = torch.exp(-log_vars)
        log_prob = -0.5 * (log_vars + (z - means)**2 * inv_var + LOG_2PI).sum(dim=2) # (batch, K)

        # weight by mixture
        weighted_log_prob = torch.log(pi.clamp_min(1e-8)) + log_prob # (batch, K)

        # log-sum-exp trick
        log_prob_sum = torch.logsumexp(weighted_log_prob, dim=1)
        
        return - (log_prob_sum + log_jacobian)


    def loss_size(self, rate, num_points):

        log_prob = num_points * torch.log(rate.clamp_min(1e-8)) - rate - torch.lgamma(num_points + 1)
        return -log_prob


    def forward(self, x, z, c, num_points):

        # forward pass
        out = self.mlp(c)

        # split output
        rate, pi, means, log_vars = self.split(out)

        rate = torch.exp(rate).squeeze(-1)
        # rate = F.softplus(rate).squeeze(-1)
        pi = F.softmax(pi, dim=1)

        loss_p = self.alpha * torch.mean(self.loss_point(pi, means, log_vars, z, x)/num_points, dim=0)
        loss_s = torch.mean(self.loss_size(rate, num_points), dim=0)

        return loss_p, loss_s
    

    def sample_num_points(self, c):

        out = self.mlp(c)
        rate, pi, means, log_vars = self.split(out)
        rate = torch.exp(rate).squeeze(-1) 

        num_points = torch.distributions.Poisson(rate).sample()
        num_points = torch.clamp(num_points, min=1).long()

        return num_points


    def sample(self, c):

        data, meta = {}, {}

        out = self.mlp(c)
        rate, pi, means, log_vars = self.split(out)
        
        rate = torch.exp(rate).squeeze(-1) 
        # rate = F.softplus(rate).squeeze(-1)
        pi = F.softmax(pi, dim=1)
    
        num_points = torch.distributions.Poisson(rate).sample()
        num_points = num_points.cpu().numpy().astype(int)

        for i in range(len(num_points)):

            if num_points[i] == 0:
                continue

            mix = torch.distributions.Categorical(pi[i])
            gauss = torch.distributions.Normal(means[i], torch.exp(0.5 * log_vars[i]))
            comp = torch.distributions.Independent(gauss, 1)
            mog = torch.distributions.MixtureSameFamily(mix, comp)
            z = mog.sample(sample_shape=(num_points[i],))

            z = z.cpu().numpy()
            
            for j, var in enumerate(self.z_vars):
                if var in data:
                    data[var] = np.concatenate((data[var], z[:, j]))
                else:
                    data[var] = z[:, j]

        c = c.cpu().numpy()        
        meta["eid"] = np.arange(len(num_points))
        for j, var in enumerate(self.c_vars):
            meta[var] = c[:, j]     
        data["eid"] = np.repeat(meta["eid"], num_points)
        
        return data, meta


class ConditionalFlowMatching(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        # auxillary model used for inference     
        self.aux_model = None
        self.aux_model_dir = cfg.aux_model_dir

        # number integration steps 
        self.num_steps = cfg.num_steps

        # variable names for z and c
        _, self.z_vars, self.c_vars = setup_var_names(cfg.transforms, cfg.spherical) 

        # input dimensions 
        self.point_dim = len(self.z_vars)
        self.cond_dim = cfg.cond_dim
        
        # neural nets        
        self.cfg_encoder = cfg.encoder
        self.encoder = self._build_encoder(self.cfg_encoder) if self.cfg_encoder is not None else None
        input_size_mlp = self._input_size_mlp()

        self.mlp = MLP(
            hidden_layers=cfg.mlp.hidden_layers, 
            layer_norm=cfg.mlp.layer_norm, 
            input_size=input_size_mlp, 
            output_size=self.point_dim,
            activation=cfg.mlp.activation,
            )


    def _build_encoder(self, cfg):

        self.use_cond = cfg.use_cond # t, c
        self.encoder_name = cfg.name

        input_size = self.point_dim
        if self.use_cond:
            input_size += self.cond_dim + 1

        encoder_cls = ENCODER_REGISTRY[self.encoder_name]            
        cfg = {k: v for k, v in vars(cfg).items() if k not in ["name", "use_cond"]}
        encoder = encoder_cls(input_size=input_size, **cfg)

        return encoder


    def _input_size_mlp(self):
        
        if self.encoder is None:
            return self.point_dim + self.cond_dim + 1

        if self.encoder_name in ["sequence"]:
            return self.encoder.output_size

        if self.encoder_name in ["deepsets", "pointnet"]:
            return self.point_dim + self.cond_dim + 1 + self.encoder.output_size

        raise ValueError(f"Unknown encoder: {self.encoder_name}")


    def z_t(self, z_0, z_1, t):
        return t * z_1 + (1-t) * z_0


    def v_t(self, z_0, z_1):
        return z_1 - z_0


    def v_theta(self, z_t, t, c, num_points):

        if self.encoder is None:
            inputs = torch.cat([z_t, t, c], dim=-1)
            return self.mlp(inputs)
        
        if self.encoder_name in ["sequence"]:
            inputs = self.encoder(z_t, t, c, num_points)
            # later we can add conditionals again v = mlp([z_t, h])

        elif self.encoder_name in ["deepsets", "pointnet"]:
            emb = self.encoder(z_t, num_points)
            inputs = torch.cat([z_t, t, c, emb], dim=-1)
        else:
            raise ValueError(self.encoder_name)

        return self.mlp(inputs)


    def forward(self, x, z, c, num_points):        

        device = z.device
        batch_size= c.size(0)
        
        c_repeated = torch.repeat_interleave(c, num_points, dim=0)

        # sample the time step per batch element
        t = torch.distributions.uniform.Uniform(0, 1).sample(sample_shape=(batch_size,)).to(z.device)
        t = torch.repeat_interleave(t.unsqueeze(-1), num_points, dim=0)

        # sample z_0 from p_0
        z_0 = torch.distributions.Normal(0, 1).sample(z.shape).to(device)

        z_t = self.z_t(z_0, z, t)
        v_t = self.v_t(z_0, z) 
        v_theta = self.v_theta(z_t, t, c_repeated, num_points)

        loss = self.loss(v_theta, v_t, num_points)

        return loss


    def loss(self, v_theta, v_t, num_points):
        # computes average point loss per point cloud and takes average across batch
        # contract: tensor is always concatenated in event order and never reordered
        
        err = ((v_theta - v_t)**2).sum(dim=-1)  # total loss per point 
        err = torch.segment_reduce(err, reduce="mean", lengths=num_points)
        
        return torch.mean(err)


    def _load_model_aux(self, device):

        # load config file of auxillary model         
        cfg_aux_model = load_config(f"{self.aux_model_dir}/config.yaml")

        # load auxillary model
        self.aux_model = MixtureDensityNetwork(cfg_aux_model.model).to(device)
        load_checkpoint(self.aux_model_dir, self.aux_model, device)


    def sample(self, c):

        device = c.device 
        batch_size = c.size(0)

        # load aux model if it does not exist
        if self.aux_model is None:
            self._load_model_aux(device)

        # sample from auxilary model
        num_points = self.aux_model.sample_num_points(c)
        c_repeated = torch.repeat_interleave(c, num_points, dim=0)   

        # sample noise
        total_points = num_points.sum().item()
        shape = (total_points, self.point_dim)
        z_t = torch.distributions.Normal(0,1).sample(shape).to(device)

        # step size   
        delta_t = torch.full((total_points, 1), 1/self.num_steps, device=device)

        # integrate 
        for i in range(self.num_steps):            
            t = (i+1)*delta_t
            v_theta = self.v_theta(z_t, t, c_repeated, num_points)
            z_t += v_theta * delta_t

        return self.to_flattened_representation(z_t, c, num_points)  


    def to_flattened_representation(self, z, c, num_points):
        
        data, meta = {}, {}

        z = z.cpu().numpy()      
        c = c.cpu().numpy()
        num_points = num_points.cpu().numpy().astype(int)  

        for j, var in enumerate(self.z_vars):
            data[var] = z[:, j]

        for j, var in enumerate(self.c_vars):
            meta[var] = c[:, j]    

        meta["eid"] = np.arange(len(num_points))
        data["eid"] = np.repeat(meta["eid"], num_points)

        return data, meta

class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "pointnet"





class DeepSetsEncoder(nn.Module):
    def __init__(
            self,
            hidden_layers,
            layer_norm,
            input_size,
            output_size,
            activation,
            pooling,
            ):

        super().__init__()

        self.phi_net = MLP(
            hidden_layers,
            layer_norm,
            input_size,
            output_size,
            activation,
        )

        self.pooling = create_pooling_fn(pooling)
        self.output_size = output_size


    def forward(self, z_t, num_points):

        phi_output = self.phi_net(z_t)
        splits = torch.split(phi_output, num_points.detach().cpu().tolist(), dim=0)

        # pooling
        pooled_list = []
    
        # change this later to avoid loop
        for chunk in splits:
            pooled_list.append(self.pooling(chunk))

        emb = torch.stack(pooled_list)
        return torch.repeat_interleave(emb, num_points, dim=0)



class SequenceEncoder(nn.Module):

    def __init__(
            self,
            max_seq_len,
            input_size,
            output_size,
            cell_type,
            ):
        
        super().__init__()

        self.max_seq_len = max_seq_len
        self.output_size = output_size
        
        if cell_type == "lstm":
            self.rnn = nn.LSTM(input_size, output_size, batch_first=True)
        elif cell_type == "gru":
            self.rnn = nn.GRU(input_size, output_size, batch_first=True)
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")


    def _to_padded(self, tensor, num_points, batch_size):

        tensor_padded = torch.zeros(batch_size, self.max_seq_len, tensor.size(-1), device=tensor.device)
        splits = torch.split(tensor, num_points.tolist(), dim=0)
    
        for i, chunk in enumerate(splits):
            tensor_padded[i, :chunk.size(0), :] = chunk
    
        return tensor_padded

    def _create_mask(self, num_points, max_seq_len, device):
        return torch.arange(max_seq_len, device=device)[None, :] < num_points[:, None]

    def forward(self, z_t, t, c, num_points):

        batch_size = num_points.size(0)
  
        z_t = self._to_padded(z_t, num_points, batch_size)
        c = self._to_padded(c, num_points, batch_size)
        t = self._to_padded(t, num_points, batch_size)

        sequence = torch.cat([z_t, t, c], dim=-1)

        packed = pack_padded_sequence(sequence, num_points.cpu(), batch_first=True, enforce_sorted=False)
        hidden_states, _ = self.rnn(packed)
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True, total_length=self.max_seq_len)

        # to sparse representation
        mask = self._create_mask(num_points, self.max_seq_len, z_t.device)

        return hidden_states[mask]


class MLP(nn.Module):
    def __init__(
            self,
            hidden_layers,
            layer_norm,
            input_size,
            output_size,
            activation,
            ):
        super().__init__()

        layers = []
        activation = ACTIVATIONS[activation]
        
        in_features = input_size

        for hidden in hidden_layers: 
            layers.append(nn.Linear(in_features, hidden))

            if layer_norm:
                layers.append(nn.LayerNorm(hidden))

            layers.append(activation()) 
            in_features = hidden

        layers.append(nn.Linear(in_features, output_size))

        self.net = nn.Sequential(*layers)


    def forward(self, inputs):
        return self.net(inputs)


def load_checkpoint(run_dir, model, device, which="best"):

    file_path = os.path.join(run_dir, f"{which}_model.pt")
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def create_pooling_fn(pooling):

    if pooling == "mean":
        return lambda x: x.mean(dim=0)
    elif pooling == "max":
        return lambda x: x.max(dim=0)[0]
    elif pooling == "sum":
        return lambda x: x.sum(dim=0) / (x.size(0) ** 0.5)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")






# class PointBatch:

#     def __init__(self):
#         # .to_padded()
#         # .to_sparse()
#         # .mask
#         # .lengths



# stochastic point subsampling

# Instead of using all points, you sample a subset:


#point net impplementation #sparse batching.
# deepsets? # sparse batching.
# set transformer 





# pointwise mlp (use fx flat data)
#cross attention
# how can graph nn be used? 


# set-based model: Deep Sets or Set Transformer
# Autoregressive / chunked MDN: Generate points in chunks of 500–1000


# This is a matching problem.

# Solutions used in research:

# Optimal transport
# Hungarian matching
# random pairing
# coupling distributions





#questions:
# Are all points part of ONE trajectory, or multiple independent points/events?
# PointNet-style (set-based)
# Transformer
# apply a point-wise net to predict velocity field of individual points. 

# cfm model  v1
# architecture: pad to a max_seq_len and use lstm network to predict velocity field using points sorted by time.
# core idea: points depend only on points that came before in time.  
# assuption: strong temporal structure

# it tries to compress everything into: hidden_t = summary of all previous hits
# so it mixes: different particle branches, unrelated hits, artificial ordering




# TO DO:
# implement sampling procedure
# test assuption by randomly permuting order of points other?
# add more linear layers to hidden output and activation (nn.SiLU())
# sliding window to avoid long term memory 


# cfm model v2 
















# input:
# z either shape(b, max_seq_len, 4) or shape(b, 4)
# t shape(b,)
# c shape (b, 4)
# possibly (b, embedding_dim)

# input(z,t,c, )

# next how to trainr

# considerations 
# batch of random points:
# then input to the model is z with shape(b, 4), c with shape(b, 4) and t with shape(b,1) and some aggregated information about
# the shower we shape(b, embedding_dim)
# output of the model is  the velocity field with shape(b, 4)
# batches contains points from different events 
# we do not have the aggregated information about the shower at inference. 
# we also need to learn an embedding somehow 





# pooling network. 

# some low dimensional representation of the entire shower (we already have c, which is kind of a low dimensional representation), but
# could add to this also a representation of an embedding of the shower, possibly we need to learn this embedding.



# predict kun første unique, predict path and the sample points along path?



MODEL_REGISTRY = {
    "mdn": MixtureDensityNetwork,
    "cfm": ConditionalFlowMatching,
}

ENCODER_REGISTRY = {
    "pointnet": PointNetEncoder,
    "deepsets": DeepSetsEncoder,
    "sequence": SequenceEncoder,
}