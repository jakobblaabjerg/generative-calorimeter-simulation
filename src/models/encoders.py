import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .blocks import MLP
from .registry import register_encoder


class TNet(torch.nn.Module):
    
    def __init__(
            self,
            input_size,
            output_size,
            mlp1_layers,
            mlp2_layers,
            activation,
            pooling,
            batch_norm,
            ):
        
        super().__init__()

        self.mlp1 = MLP(
            hidden_layers=mlp1_layers[:-1],
            batch_norm=batch_norm,
            input_size=input_size,
            output_size=mlp1_layers[-1],
            activation=activation,
        )

        self.mlp2 = MLP(
            hidden_layers=mlp2_layers,
            batch_norm=batch_norm,
            input_size=mlp1_layers[-1],
            output_size=output_size,
            activation=activation,
        )

        assert input_size**2 == output_size

        self.input_size = input_size
        self.pooling = Pooling(method=pooling)

        # initialize last layer
        with torch.no_grad():
            self.mlp2.last.weight.zero_()
            identity = torch.eye(input_size)
            self.mlp2.last.bias.copy_(identity.view(-1))

    def forward(self, inputs, num_points):

        batch_size = num_points.size(0)
        mlp1_output = self.mlp1(inputs)
        splits = torch.split(mlp1_output, num_points.tolist(), dim=0)
        pooled = torch.stack([self.pooling(s) for s in splits])
        mlp2_output = self.mlp2(pooled)
        T = mlp2_output.view(batch_size, self.input_size, self.input_size)

        T_repeated= torch.repeat_interleave(T, num_points, dim=0)
        input_transformed = torch.bmm(T_repeated, inputs.unsqueeze(-1)).squeeze(-1) 

        return input_transformed, T

@register_encoder("pointnet") 
class PointNetEncoder(torch.nn.Module):
    
    def __init__(
            self,
            input_size,
            activation,
            pooling,
            batch_norm,
            mlp1_layers,
            mlp2_layers,
            tnet_layers,
            lambda_reg,
            dropout_rate
            ):

        super().__init__()
        self.name = "pointnet"
        self.lambda_reg = lambda_reg
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.output_size = mlp2_layers[-1]

        self.input_transform = TNet(
            input_size=input_size,
            output_size=input_size*input_size,
            mlp1_layers=tnet_layers[0],
            mlp2_layers=tnet_layers[1],
            activation=activation,
            pooling=pooling,
            batch_norm=batch_norm
        )

        self.mlp1 = MLP(
            hidden_layers=mlp1_layers[:-1],
            input_size=input_size,
            activation=activation,
            output_size=mlp1_layers[-1],
            batch_norm=batch_norm
        )

        self.feature_transform = TNet(
            input_size=mlp1_layers[-1],
            output_size=mlp1_layers[-1]*mlp1_layers[-1],
            mlp1_layers=tnet_layers[0],
            mlp2_layers=tnet_layers[1],
            activation=activation,
            pooling=pooling,
            batch_norm=batch_norm
        )

        self.mlp2 = MLP(
            hidden_layers=mlp2_layers[:-1],
            input_size=mlp1_layers[-1], 
            activation=activation,
            output_size=mlp2_layers[-1],
            batch_norm=batch_norm
        )

        self.pooling = Pooling(method=pooling)


    def forward(self, z_t, t, c, num_points):

        input_transformed, T_input = self.input_transform(z_t, num_points)
        mlp1_output = self.mlp1(input_transformed)

        feature_transformed, T_feature = self.feature_transform(mlp1_output, num_points)
        mlp2_output = self.mlp2(feature_transformed)

        reg_loss = self.lambda_reg * (self.tnet_reg_loss(T_input) + self.tnet_reg_loss(T_feature))

        # pooling
        splits = torch.split(mlp2_output, num_points.tolist(), dim=0)
        pooled = torch.stack([self.pooling(s) for s in splits]) 
        
        # dropout
        pooled = self.dropout(pooled)
        
        # global feature
        emb = torch.repeat_interleave(pooled, num_points, dim=0)

        return torch.cat([z_t, t, c, emb], dim=-1), reg_loss


    def tnet_reg_loss(self, T):
        I = torch.eye(T.size(1), device=T.device).unsqueeze(0)
        loss = torch.norm(torch.bmm(T, T.transpose(1, 2)) - I, dim=(1,2))
        return loss.mean()

@register_encoder("deepsets")
class DeepSetsEncoder(torch.nn.Module):
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
            hidden_layers=hidden_layers,
            layer_norm=layer_norm,
            input_size=input_size,
            output_size=output_size,
            activation=activation,
        )

        self.pooling = Pooling(method=pooling)
        self.output_size = output_size

    def forward(self, z_t, t, c, num_points):
        phi_output = self.phi_net(z_t)
        splits = torch.split(phi_output, num_points.tolist(), dim=0)
        pooled = torch.stack([self.pooling(s) for s in splits]) 

        emb = torch.repeat_interleave(pooled, num_points, dim=0)
        
        return torch.cat([z_t, t, c, emb], dim=-1), torch.tensor(0.0, device=z_t.device)

@register_encoder("sequence")  
class SequenceEncoder(torch.nn.Module):

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
            self.rnn = torch.nn.LSTM(input_size, output_size, batch_first=True)
        elif cell_type == "gru":
            self.rnn = torch.nn.GRU(input_size, output_size, batch_first=True)
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")


    def _to_padded(self, tensor, num_points, batch_size):

        tensor_padded = torch.zeros(batch_size, self.max_seq_len, tensor.size(-1), device=tensor.device)
        splits = torch.split(tensor, num_points.tolist(), dim=0)
    
        for i, chunk in enumerate(splits):
            length = min(chunk.size(0), self.max_seq_len)
            tensor_padded[i, :length, :] = chunk[:length]
    
        return tensor_padded

    def _create_mask(self, num_points, max_seq_len, device):
        return torch.arange(max_seq_len, device=device)[None, :] < num_points[:, None]

    def forward(self, z_t, t, c, num_points):

        batch_size = num_points.size(0)
  
        z_t = self._to_padded(z_t, num_points, batch_size)
        c = self._to_padded(c, num_points, batch_size)
        t = self._to_padded(t, num_points, batch_size)

        sequence = torch.cat([z_t, t, c], dim=-1)

        num_points_trunc = torch.clamp(num_points, max=self.max_seq_len)

        packed = pack_padded_sequence(sequence, num_points_trunc.cpu(), batch_first=True, enforce_sorted=False)
        hidden_states, _ = self.rnn(packed)
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True, total_length=self.max_seq_len)

        # to sparse representation
        mask = self._create_mask(num_points_trunc, self.max_seq_len, z_t.device)

        # later we can add conditionals again v = mlp([z_t, h])
        return hidden_states[mask], torch.tensor(0.0, device=z_t.device)



class Pooling(torch.nn.Module):

    def __init__(self, method):

        super().__init__()

        self.method = method

    def forward(self, x):

        if self.method == "mean":
            return x.mean(dim=0)

        elif self.method == "max":
            return x.max(dim=0)[0]

        elif self.method == "sum":
            return x.sum(dim=0) / x.size(0)**0.5

        raise ValueError(f"Unknown pooling: {self.method}")