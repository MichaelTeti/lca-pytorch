from math import sqrt

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


Tensor = torch.Tensor


def _make_feature_grid_1D(weights: Tensor, scale_each: bool = False) -> Tensor:
    assert len(weights.shape) == 3
    if weights.shape[1] == 1:
        return weights[:, 0]
    else:
        nrow = int(sqrt(weights.shape[0]))
        feat_grid = make_grid(
            weights.unsqueeze(1),
            nrow,
            normalize=True,
            scale_each=scale_each,
            pad_value=0.5,
        )
        return feat_grid[0]


def _make_feature_grid_2D(weights: Tensor, scale_each: bool = False) -> Tensor:
    assert len(weights.shape) == 4
    nrow = int(sqrt(weights.shape[0]))
    feat_grid = make_grid(
        weights, nrow, normalize=True, scale_each=scale_each, pad_value=0.5
    )
    return feat_grid.permute(1, 2, 0)


def _make_feature_grid_3D(weights: Tensor, scale_each: bool = False) -> Tensor:
    assert len(weights.shape) == 5
    nrow = int(sqrt(weights.shape[0]))
    T = weights.shape[2]
    feat_grids = []
    for t in range(T):
        grid_t = make_grid(
            weights[:, :, t], nrow, normalize=True, scale_each=scale_each, pad_value=0.5
        )
        feat_grids.append(grid_t)
    return torch.stack(feat_grids).permute(0, 2, 3, 1)


def make_feature_grid(weights: Tensor, scale_each: bool = False) -> Tensor:
    if len(weights.shape) == 3:
        return _make_feature_grid_1D(weights, scale_each)
    elif len(weights.shape) == 4:
        return _make_feature_grid_2D(weights, scale_each)
    elif len(weights.shape) == 5:
        return _make_feature_grid_3D(weights, scale_each)
