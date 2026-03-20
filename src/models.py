import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "leaky_relu": nn.LeakyReLU
}

class GaussianMixtureModel(nn.Module):

    def __init__(self, 
                 input_dim, 
                 output_dim,
                 K,
                 hidden_layers=None, 
                 activation="relu",
                 layer_norm=False,
                 ):

        super().__init__()

        self.K = K 
        self.D = output_dim

        layers = []
        in_features = input_dim
        activation = ACTIVATIONS[activation]

        for hidden in hidden_layers: 
            layers.append(nn.Linear(in_features, hidden))

            if layer_norm:
                layers.append(nn.LayerNorm(hidden))

            layers.append(activation()) 
            in_features = hidden

        layers.append(nn.Linear(in_features, K * (2 * output_dim + 1)))

        self.net = nn.Sequential(*layers)


    def forward(self, c):

        batch_size = c.size(0)
        out = self.net(c)

        # split output
        pi = out[:, :self.K]
        mu = out[:, self.K:self.K+self.K*self.D]
        log_var = out[:, self.K+self.K*self.D:]
    
        mu = mu.view(batch_size, self.K, self.D)
        log_var = log_var.view(batch_size, self.K, self.D)

        pi = F.softmax(pi, dim=1)

        return pi, mu, log_var