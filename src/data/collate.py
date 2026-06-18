import torch
from functools import partial


def collate_padded(batch, max_seq_len):

    """
    Collate function for variable-length sequences with padding.
    Each sample is truncated or padded to `max_seq_len`.

    Parameters
    ----------
    batch : list of tuples
        Each element is (x, z, c), where:
        - x: (N_i, point_dim) tensor
        - z: (N_i, point_dim) tensor
        - c: (1, cond_dim) tensor
    max_seq_len : int
        Maximum sequence length to pad/truncate to.

    Returns
    -------
    x_padded : torch.Tensor
        Shape (B, max_seq_len, point_dim)
    z_padded : torch.Tensor
        Shape (B, max_seq_len, point_dim)
    c : torch.Tensor
        Shape (B, cond_dim)
    num_points : torch.Tensor
        Number of valid points per sample (B,)
    """

    x_list, z_list, c_list = zip(*batch)

    batch_size = len(x_list)
    point_dim = z_list[0].size(1)

    num_points = [min(x.size(0), max_seq_len) for x in x_list]

    x_padded = torch.zeros(batch_size, max_seq_len, point_dim, dtype=x_list[0].dtype)
    z_padded = torch.zeros(batch_size, max_seq_len, point_dim, dtype=z_list[0].dtype)

    for i in range(batch_size):
        x_padded[i, :num_points[i]] = x_list[i][:num_points[i]]
        z_padded[i, :num_points[i]] = z_list[i][:num_points[i]]

    num_points = torch.tensor(num_points, dtype=torch.long)
    c = torch.stack(c_list)

    return x_padded, z_padded, c, num_points


def collate_sparse(batch):

    """
    Collate function for sparse (concatenated) representation.
    All samples are flattened into a single tensor batch.

    Parameters
    ----------
    batch : list of tuples
        Each element is (x, z, c), where:
        - x: (N_i, point_dim)
        - z: (N_i, point_dim)
        - c: (1, cond_dim)

    Returns
    -------
    x : torch.Tensor
        Shape (sum N_i, point_dim)
    z : torch.Tensor
        Shape (sum N_i, point_dim)
    c : torch.Tensor
        Shape (B, cond_dim)
    num_points : torch.Tensor
        Number of points per sample (B,)
    """

    x_list, z_list, c_list = zip(*batch)

    num_points = torch.tensor([x.size(0) for x in x_list], dtype=torch.long)

    x = torch.cat(x_list, dim=0)
    z = torch.cat(z_list, dim=0)
    c = torch.stack(c_list)

    return x, z, c, num_points


def collate_index(store):

    """
    Factory for index-based collate function.
    Creates a collate function that retrieves samples directly
    from a pre-stored tensor container using indices.

    Parameters
    ----------
    store : TensorStore
        Preloaded tensors (x, z, c, num_points)

    Returns
    -------
    function
        Collate function of signature:
        collate(indices) -> (x, z, c, num_points)
    """

    def collate_fn(indices):
        
        idxs = torch.as_tensor(indices, dtype=torch.long)
        
        return (
            store.x.index_select(0, idxs),
            store.z.index_select(0, idxs),
            store.c.index_select(0, idxs),
            store.num_points.index_select(0, idxs)
        )
    return collate_fn


def create_collate_fn(mode, **kwargs):
    """
    Returns a PyTorch collate function based on registry.
    """

    if mode not in COLLATE_REGISTRY:
        raise ValueError(f"Unknown mode: {mode}")

    return COLLATE_REGISTRY[mode](**kwargs)


COLLATE_REGISTRY = {
    "sparse": lambda **kwargs: collate_sparse,
    "padded": lambda max_seq_len, **kwargs: partial(collate_padded, max_seq_len=max_seq_len),
    "index": lambda store, **kwargs: collate_index(store=store),
}

