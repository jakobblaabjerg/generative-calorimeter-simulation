import torch

def filter_dict(data, mask):
    return {k: v[mask] for k, v in data.items()}


def create_key(name, existing_keys):

    suffixes = []
    
    for key in existing_keys:
        if key.startswith(f"{name}_"):
            suffixes.append(int(key.split("_")[-1]))
    
    return f"{name}_{max(suffixes, default=0) + 1}"


def set_seed(seed=0):
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_cuda(device):

    if device.type == "cuda":
        torch.cuda.synchronize()


def move_to_device(batch, device):
    
    if torch.is_tensor(batch):
        return batch.to(device)
    
    if isinstance(batch, (tuple, list)):
        return type(batch)(x.to(device) if torch.is_tensor(x) else x for x in batch)
    
    return batch