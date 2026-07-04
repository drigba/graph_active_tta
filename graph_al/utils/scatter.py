import torch
from torch import Tensor


def scatter_sum(src: Tensor, index: Tensor, dim: int = 0, dim_size: int | None = None) -> Tensor:
    if dim < 0:
        dim += src.dim()
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = src.new_zeros(out_shape)
    index = index.to(src.device)
    view = [1] * src.dim()
    view[dim] = -1
    index = index.view(view).expand_as(src)
    return out.scatter_add_(dim, index, src)
