import torch, math
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 
import os

from src.data_processing import safe_underscore

LOG_2PI = math.log(2 * math.pi)

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "leaky_relu": nn.LeakyReLU
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
    # conditional Mixture Density Network (MDN) with Poisson multiplicity

    def __init__(self,
                 k,
                 input_dim,
                 hidden_layers,
                 transforms,
                 add_jacobian,
                 spherical=False, 
                 activation="relu",
                 layer_norm=False,
                 ):

        super().__init__()

        self.k = k
       
        if add_jacobian:
            self.z_hat_jac_func = make_jac_func(transforms.z_hat)
            self.e_jac_func = make_jac_func(transforms.e)
       
        transform_e = safe_underscore(transforms.e)
        transform_z_hat = safe_underscore(transforms.z_hat)

        if spherical:
            self.D = 3
            self.data_cols = ["r_hat_norm", f"z_hat{transform_z_hat}_norm", f"e{transform_e}_norm"]
        else:
            self.D = 4
            self.data_cols = ["x_hat_norm", "y_hat_norm", f"z_hat{transform_z_hat}_norm", f"e{transform_e}_norm"]
        self.meta_cols = ["dir_x_norm", "dir_y_norm", "dir_z_norm", "e_inc_norm"]


        layers = []
        activation = ACTIVATIONS[activation]
        
        in_features = input_dim

        for hidden in hidden_layers: 
            layers.append(nn.Linear(in_features, hidden))

            if layer_norm:
                layers.append(nn.LayerNorm(hidden))

            layers.append(activation()) 
            in_features = hidden

        layers.append(nn.Linear(in_features, 1 + self.k * (2 * self.D + 1)))

        self.net = nn.Sequential(*layers)


    def split(self, out):

        batch_size = out.size(0)

        idx = 0

        rate = out[:, idx:idx+1]
        idx += 1

        pi = out[:, idx:idx+self.k]
        idx += self.k

        means = out[:, idx:idx+self.k*self.D]
        means = means.view(batch_size, self.k, self.D)
        idx += self.k*self.D

        log_vars = out[:, idx:idx+self.k*self.D]
        log_vars = log_vars.view(batch_size, self.k, self.D)

        return rate, pi, means, log_vars
  

    def loss_point(self, pi, means, log_vars, z, x):

        log_jacobian = torch.zeros_like(x)        

        if self.e_jac_func is not None:
            log_jacobian[:, -1] = self.e_jac_func(x[:, -1])
        if self.z_hat_jac_func is not None:
            log_jacobian[:, -2] = self.z_hat_jac_func(x[:, -2])

        log_jacobian = log_jacobian.sum(dim=1)

        _, K, _ = means.shape
        z = z.unsqueeze(1).expand(-1, K, -1)  # (batch, K, D)
        
        # gaussian log probability
        inv_var = torch.exp(-log_vars)
        log_prob = -0.5 * (log_vars + (z - means)**2 * inv_var + LOG_2PI).sum(dim=2) # (batch, K)

        # weight by mixture
        weighted_log_prob = torch.log(pi.clamp_min(1e-8)) + log_prob # (batch, K)

        # log-sum-exp trick
        log_prob_sum = torch.logsumexp(weighted_log_prob, dim=1)
        
        return - (log_prob_sum + log_jacobian)


    def loss_count(self, rate, N):

        log_prob = N * torch.log(rate.clamp_min(1e-8)) - rate - torch.lgamma(N + 1)
        return -log_prob


    def forward(self, z, N, c, x):

        # forward pass
        out = self.net(c)

        # split output
        rate, pi, means, log_vars = self.split(out)

        rate = torch.exp(rate).squeeze(-1)
        # rate = F.softplus(rate).squeeze(-1)
        pi = F.softmax(pi, dim=1)

        alpha = 10000

        lp = alpha * torch.mean(self.loss_point(pi, means, log_vars, z, x)/N, dim=0)
        lc = torch.mean(self.loss_count(rate, N), dim=0)

        return lp, lc


    def sample(self, c):

        data, meta = {}, {}

        out = self.net(c)
        rate, pi, means, log_vars = self.split(out)
        
        rate = torch.exp(rate).squeeze(-1) 
        # rate = F.softplus(rate).squeeze(-1)
        pi = F.softmax(pi, dim=1)
    
        N = torch.distributions.Poisson(rate).sample()
        N = N.cpu().numpy().astype(int)

        for i in range(len(N)):

            if N[i] == 0:
                continue

            mix = torch.distributions.Categorical(pi[i])
            gauss = torch.distributions.Normal(means[i], torch.exp(0.5 * log_vars[i]))
            comp = torch.distributions.Independent(gauss, 1)
            mog = torch.distributions.MixtureSameFamily(mix, comp)
            x = mog.sample(sample_shape=(N[i],))

            x = x.cpu().numpy()

            for j, col in enumerate(self.data_cols):
                
                if col in data:
                    data[col] = np.concatenate((data[col], x[:, j]))
                else:
                    data[col] = x[:, j]

        
        c = c.cpu().numpy()        
        meta["eid"] = np.arange(len(N))
        
        for j, col in enumerate(self.meta_cols):
            meta[col] = c[:, j]     

        data["eid"] = np.repeat(meta["eid"], N)
        
        return data, meta


def load_checkpoint(run_dir, model, device, which="best"):

    file_path = os.path.join(run_dir, f"{which}_model.pt")
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
