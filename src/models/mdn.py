import math
import torch
import torch.nn.functional as F
import numpy as np 

from .blocks import MLP
from .base import BaseModel
from .registry import register_model


from src.data.datasets import create_var_names 
from src.calosim import CaloSimDataset

LOG_2PI = math.log(2 * math.pi)


def create_jacobian(transform: str):

    if transform == "":
        return None
    elif transform == "sqrt":
        coef = -0.5
    elif transform == "log":
        coef = -1.0
    else:
        raise ValueError(f"Unknown transform {transform}")

    return lambda x: coef * torch.log(x.clamp_min(1e-8))



class BaseMDN(BaseModel):

    def __init__(self, cfg):

        super().__init__()

        self.k = cfg.k
        self.alpha = cfg.alpha

        self.add_jacobian = cfg.add_jacobian
       
        if self.add_jacobian:
            self.z_hat_jacobian = create_jacobian(cfg.transforms.z_hat)
            self.e_jacobian = create_jacobian(cfg.transforms.e)

        self.x_vars, self.z_vars, self.c_vars = create_var_names(cfg.input_vars, cfg.transforms)
        self.point_dim = len(self.z_vars)
        self.cond_dim = len(self.c_vars)

        self.idx_e = self.x_vars.index("e")
        self.idx_z_hat = self.x_vars.index("z_hat")


    def to_dataset(self, z, c, num_points):

        data, meta = {}, {}

        for i in range(len(z)):

            for j, var in enumerate(self.z_vars):
                if var in data:
                    data[var] = np.concatenate((data[var], z[i][:, j]))
                else:
                    data[var] = z[i][:, j]

        c = c.cpu().numpy()        
        meta["idx"] = np.arange(len(num_points))

        for j, var in enumerate(self.c_vars):
            meta[var] = c[:, j]     

        data["idx"] = np.repeat(meta["idx"], num_points)
        
        return CaloSimDataset(data=data, meta=meta)

    @torch.no_grad()
    def sample_mixture(self, pi, mean, log_var, num_points):

        mix = torch.distributions.Categorical(pi)
        gauss = torch.distributions.Normal(mean, torch.exp(0.5 * log_var))
        comp = torch.distributions.Independent(gauss, 1)
        mog = torch.distributions.MixtureSameFamily(mix, comp)
        z = mog.sample(sample_shape=(num_points,))

        z = z.cpu().numpy()

        return z


@register_model("mdnV1")
class MixtureDensityNetworkV1(BaseMDN):

    def __init__(self, cfg):

        super().__init__(cfg)

        self.mlp = MLP(
            hidden_layers=cfg.mlp.hidden_layers, 
            layer_norm=cfg.mlp.layer_norm, 
            input_size=self.cond_dim, 
            output_size=1+self.k*(2*self.point_dim+1), 
            activation=cfg.mlp.activation
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

        log_jacobian = torch.zeros(x.size(0), device=x.device)        

        if self.add_jacobian and self.e_jacobian is not None:
            log_jacobian += self.e_jacobian(x[:, self.idx_e])
        if self.add_jacobian and self.z_hat_jacobian is not None:
            log_jacobian += self.z_hat_jacobian(x[:, self.idx_z_hat])

        z = z.unsqueeze(1).expand(-1, self.k, -1)  # (batch, k, point_dim)
        
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

        loss_point = self.alpha * torch.mean(self.loss_point(pi, means, log_vars, z, x)/num_points, dim=0)
        loss_size = torch.mean(self.loss_size(rate, num_points), dim=0)

        return loss_point, loss_size
    

    def sample_num_points(self, c):

        out = self.mlp(c)
        rate, pi, means, log_vars = self.split(out)
        rate = torch.exp(rate).squeeze(-1) 

        num_points = torch.distributions.Poisson(rate).sample()
        num_points = torch.clamp(num_points, min=1).long()

        return num_points

    @torch.no_grad()
    def sample(self, c):

        out = self.mlp(c)
        rate, pi, means, log_vars = self.split(out)
        
        rate = torch.exp(rate).squeeze(-1) 
        pi = F.softmax(pi, dim=1)    
        num_points = torch.distributions.Poisson(rate).sample()
        num_points = num_points.cpu().numpy().astype(int)
        num_points = np.maximum(num_points, 1)

        z = []

        for i in range(len(num_points)):
           
            z.append(self.sample_mixture(pi[i], means[i], log_vars[i], num_points[i]))

        return self.to_dataset(z, c, num_points)

@register_model("mdnV2")
class MixtureDensityNetworkV2(BaseMDN):

    def __init__(self, cfg):

        super().__init__(cfg)
      
        self.mlp = MLP(
            hidden_layers=cfg.mlp.hidden_layers, 
            layer_norm=cfg.mlp.layer_norm, 
            input_size=self.cond_dim, 
            output_size=cfg.mlp.output_size,
            activation=cfg.mlp.activation
            )

        self.poisson_head = MLP(
            hidden_layers=cfg.poisson_head.hidden_layers,
            layer_norm=cfg.poisson_head.layer_norm,
            input_size=cfg.mlp.output_size,
            output_size=1,
            activation=cfg.mlp.activation
        )

        self.mdn_head = MLP(
            hidden_layers=cfg.mdn_head.hidden_layers,
            layer_norm=cfg.mdn_head.layer_norm,
            input_size=cfg.mlp.output_size,
            output_size=self.k*(2*self.point_dim+1),
            activation=cfg.mlp.activation
        )

    def forward(self, x, z, c, num_points):

        # shared encoder forward pass
        emb = self.mlp(c)

        rate = self.poisson_head(emb)
        rate = torch.exp(rate).squeeze(-1)
        # rate = F.softplus(rate).squeeze(-1)
        loss_size = self.loss_size(rate, num_points)

        pi, means, log_vars = self.split(self.mdn_head(emb))        
        pi = F.softmax(pi, dim=1)
        loss_point = self.loss_point(pi, means, log_vars, z, x, num_points)

        return self.alpha * torch.mean(loss_point), torch.mean(loss_size)


    def split(self, out):

        batch_size = out.size(0)
        idx = 0

        pi = out[:, idx:idx+self.k]
        idx += self.k

        means = out[:, idx:idx+self.k*self.point_dim]
        means = means.view(batch_size, self.k, self.point_dim)
        idx += self.k*self.point_dim

        log_vars = out[:, idx:idx+self.k*self.point_dim]
        log_vars = log_vars.view(batch_size, self.k, self.point_dim)

        return pi, means, log_vars
  

    def loss_size(self, rate, num_points):

        log_prob = num_points * torch.log(rate.clamp_min(1e-8)) - rate - torch.lgamma(num_points + 1)        
        return -log_prob


    def loss_point(self, pi, means, log_vars, z, x, num_points):

        # expand inputs to shape(sum(num_points), k) to match point-level
        pi = torch.repeat_interleave(pi, num_points, dim=0)  
        means = torch.repeat_interleave(means, num_points, dim=0) 
        log_vars = torch.repeat_interleave(log_vars, num_points, dim=0)

        z = z.unsqueeze(1).expand(-1, self.k, -1)  # (sum(num_points), k, point_dim)
        
        # gaussian log probability
        inv_var = torch.exp(-log_vars)
        log_prob = -0.5 * (log_vars + (z - means)**2 * inv_var + LOG_2PI).sum(dim=2) # (sum(num_points), k)

        # weight by mixture
        weighted_log_prob = torch.log(pi.clamp_min(1e-8)) + log_prob # (sum(num_points), k)

        # log-sum-exp trick
        log_prob_sum = torch.logsumexp(weighted_log_prob, dim=1) # (sum(num_points),)
        log_jacobian = torch.zeros_like(log_prob_sum) # (sum(num_points),)

        if self.add_jacobian and self.e_jacobian is not None:
            log_jacobian += self.e_jacobian(x[:, self.idx_e])
        
        if self.add_jacobian and self.z_hat_jacobian is not None:
            log_jacobian += self.z_hat_jacobian(x[:, self.idx_z_hat])

        # normalize loss event-level
        loss_point = torch.segment_reduce(log_prob_sum+log_jacobian, reduce="mean", lengths=num_points, axis=0)        
        return - loss_point

    @torch.no_grad()
    def sample_num_points(self, c):

        emb = self.mlp(c)
        rate = self.poisson_head(emb)
        rate = torch.exp(rate).squeeze(-1) 
        num_points = torch.distributions.Poisson(rate).sample()
        num_points = torch.clamp(num_points, min=1).long()

        return num_points

    @torch.no_grad()
    def sample(self, c):

        emb = self.mlp(c)
        rate = self.poisson_head(emb)
        rate = torch.exp(rate).squeeze(-1) 
        # rate = F.softplus(rate).squeeze(-1)
        num_points = torch.distributions.Poisson(rate).sample()
        num_points = num_points.cpu().numpy().astype(int)
        num_points = np.maximum(num_points, 1)

        pi, means, log_vars = self.split(self.mdn_head(emb))
        pi = F.softmax(pi, dim=1)

        z = []

        for i in range(len(num_points)):
            z.append(self.sample_mixture(pi[i], means[i], log_vars[i], num_points[i]))

        return self.to_dataset(z, c, num_points)
    

@register_model("mdnV3")
class MixtureDensityNetworkV3(BaseModel):

    def __init__(self, cfg):

        super().__init__()

        self.feature_dim = cfg.feature_dim
        self.k = cfg.k
        self.add_jacobian = cfg.add_jacobian
       
        if self.add_jacobian:
            self.e_jacobian = create_jacobian(cfg.transforms.e)

        self.x_vars, self.z_vars, self.c_vars = create_var_names(cfg.input_vars, cfg.transforms) 
        self.cond_dim = len(self.c_vars)

        self.mlp = MLP(
            hidden_layers=cfg.mlp.hidden_layers, 
            layer_norm=cfg.mlp.layer_norm, 
            input_size=self.cond_dim, 
            output_size=cfg.mlp.output_size,
            activation=cfg.mlp.activation
            )

        self.mdn_head = MLP(
            hidden_layers=cfg.mdn_head.hidden_layers,
            layer_norm=cfg.mdn_head.layer_norm,
            input_size=cfg.mlp.output_size,
            output_size=self.k*(2*self.feature_dim+1),
            activation=cfg.mlp.activation
        )


    def forward(self, x, z, c):

        emb = self.mlp(c)
        pi, means, log_vars = self.split(self.mdn_head(emb))        
        pi = F.softmax(pi, dim=1)
        
        loss_point, loss_jacobian = self.loss_point(pi, means, log_vars, z, x)

        return torch.mean(loss_point), torch.mean(loss_jacobian)


    def split(self, out):

        batch_size = out.size(0)
        idx = 0

        pi = out[:, idx:idx+self.k]
        idx += self.k

        means = out[:, idx:idx+self.k*self.feature_dim]
        means = means.view(batch_size, self.k, self.feature_dim)
        idx += self.k*self.feature_dim

        log_vars = out[:, idx:idx+self.k*self.feature_dim]
        log_vars = log_vars.view(batch_size, self.k, self.feature_dim)

        return pi, means, log_vars
  


    def loss_point(self, pi, means, log_vars, z, x):

        z = z.unsqueeze(1).expand(-1, self.k, -1)  # (b, k, feature_dim)
        
        # gaussian log probability
        inv_var = torch.exp(-log_vars)
        log_prob = -0.5 * (log_vars + (z - means)**2 * inv_var + LOG_2PI).sum(dim=2) # (b, k)

        # weight by mixture
        weighted_log_prob = torch.log(pi.clamp_min(1e-8)) + log_prob # (b, k)

        # log-sum-exp trick
        log_prob_sum = torch.logsumexp(weighted_log_prob, dim=1) # (b,)
        log_jacobian = torch.zeros_like(log_prob_sum) # (b,)

        if self.add_jacobian and self.e_jacobian is not None:
            log_jacobian += self.e_jacobian(x).sum(dim=1)
        
        return -log_prob_sum, -log_jacobian

    @torch.no_grad()
    def sample(self, c):

        emb = self.mlp(c)

        pi, means, log_vars = self.split(self.mdn_head(emb))
        pi = F.softmax(pi, dim=1)

        z = []

        for i in range(len(c)):
            z.append(self.sample_mixture(pi[i], means[i], log_vars[i]))

        return self.to_dataset(z, c)
    

    def to_dataset(self, z, c):

        data, meta = {}, {}

        for var in self.z_vars: 
            data[var] = np.stack(z, axis=0).reshape(-1)

        c = c.cpu().numpy()        
        meta["idx"] = np.arange(len(c))

        for j, var in enumerate(self.c_vars):
            meta[var] = c[:, j]     

        data["idx"] = np.repeat(meta["idx"], self.feature_dim)
        
        return CaloSimDataset(data=data, meta=meta)


    @torch.no_grad()
    def sample_mixture(self, pi, mean, log_var):

        mix = torch.distributions.Categorical(pi)
        gauss = torch.distributions.Normal(mean, torch.exp(0.5 * log_var))
        comp = torch.distributions.Independent(gauss, 1)
        mog = torch.distributions.MixtureSameFamily(mix, comp)
        z = mog.sample(sample_shape=())

        z = z.cpu().numpy()

        return z