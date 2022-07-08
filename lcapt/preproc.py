import torch


Tensor = torch.Tensor


def zero_mean(batch: Tensor) -> Tensor:
    if len(batch.shape) == 3:
        dims = -1
    elif len(batch.shape) in [4, 5]:
        dims = (-2, -1)
    else:
        raise NotImplementedError
    
    mean = batch.mean(dim=dims, keepdim=True)
    return batch - mean


def contrast_norm(batch: Tensor, eps: float = 1e-8) -> Tensor:
    if len(batch.shape) == 3:
        dims = -1
    elif len(batch.shape) in [4, 5]:
        dims = (-2, -1)
    else:
        raise NotImplementedError

    std = batch.std(dim=dims, keepdim=True)
    return batch / (std + eps)
