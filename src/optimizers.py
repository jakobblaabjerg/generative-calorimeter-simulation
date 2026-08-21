import torch

def create_optimizer(model, config):

    if config.type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.base_lr,
            # weight_decay=getattr(config, "weight_decay", 0.0)
        )

    if config.type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.base_lr
        )

    if config.type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.base_lr,
            # weight_decay=getattr(config, "weight_decay", 0.0)
        )

    if config.type == "radam":
        return torch.optim.RAdam(
            model.parameters(),
            lr=config.base_lr
        )

    raise ValueError(f"Unknown optimizer: {config.type}")