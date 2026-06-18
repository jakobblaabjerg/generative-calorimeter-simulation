import torch.nn as nn
from .activations import ACTIVATIONS

class MLP(nn.Module):
    def __init__(
            self,
            hidden_layers,
            input_size,
            activation,
            output_size=None,
            layer_norm=False,
            batch_norm=False,
            dropout=0.0,
            ):
        super().__init__()

        layers = []
        activation = ACTIVATIONS[activation]
        
        in_features = input_size

        for hidden in hidden_layers: 
            linear = nn.Linear(in_features, hidden)
            layers.append(linear)

            if layer_norm:
                layers.append(nn.LayerNorm(hidden))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))

            layers.append(activation()) 

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_features = hidden

        if output_size is not None:
            linear = nn.Linear(in_features, output_size)
            layers.append(linear)

        self.net = nn.Sequential(*layers)
        self.last = linear  


    def forward(self, inputs):
        return self.net(inputs)