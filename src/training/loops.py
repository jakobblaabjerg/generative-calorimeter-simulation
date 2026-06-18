from tqdm import tqdm
import torch
from contextlib import nullcontext

from src.utils import move_to_device


def run_step(model, batch, optimizer=None):

    device = model.device
    batch = move_to_device(batch, device)
    loss_b = model(*batch)

    loss_b = loss_b if isinstance(loss_b, (tuple, list)) else (loss_b,)
    loss_total = sum(loss_b)

    if optimizer is not None:
        optimizer.zero_grad() 
        loss_total.backward()
        optimizer.step()

    # detach from computational graph
    loss_b = [l.detach().item() for l in loss_b]

    return loss_b


def run_epoch(model, loader, optimizer=None, desc="", postfix_key="loss"):

    iterator = tqdm(loader, desc=desc, leave=False)
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    grad_context = torch.no_grad() if not is_train else nullcontext()
    loss_total = None

    with grad_context:

        for batch in iterator:

            loss_b = run_step(model, batch, optimizer)

            if loss_total is None:
                loss_total = [0.0] * len(loss_b)

            for i, l in enumerate(loss_b):
                loss_total[i] += l
    
            iterator.set_postfix({postfix_key: sum(loss_b)})


    # normalize loss
    num_batches = len(loader)
    loss_total = [l/(num_batches) for l in loss_total] # biased estimate

    return loss_total