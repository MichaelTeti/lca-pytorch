import torch


Tensor = torch.Tensor


def standardize_inputs(batch: Tensor, eps: float = 1e-8) -> Tensor:
    """Standardize each sample in x"""
    if len(batch.shape) == 3:
        dims = -1
    elif len(batch.shape) in [4, 5]:
        dims = (-2, -1)
    else:
        raise NotImplementedError
    mean = batch.mean(dim=dims, keepdim=True)
    std = batch.std(dim=dims, keepdim=True)
    return (batch - mean) / (std + eps)
