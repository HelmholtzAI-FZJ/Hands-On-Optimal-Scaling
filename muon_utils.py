from typing import cast, List, Tuple

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.optim.optimizer import (
    _device_dtype_check_for_fused,
    _get_scalar_dtype,
)


def zeropower_via_svd(G, **kwargs):
    original_dtype = G.dtype
    G = G.to(torch.float32)
    # SVD does not support bfloat16
    if G.size(0) > G.size(1):
        G = G.T
        transpose = True
    else:
        transpose = False
    U, S, V = G.svd()
    X = U @ V.T
    if transpose:
        X = X.T
    return X.to(original_dtype).contiguous()


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """

    assert (
        len(G.shape) == 2
    ), f"Please make sure gradients are 2D tensors to use NS, got shape: {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    #     for a, b, c in [ # updated coefficients from @leloykun
    #     (4.0848, -6.8946, 2.9270),
    #     (3.9505, -6.3029, 2.6377),
    #     (3.7418, -5.5913, 2.3037),
    #     (2.8769, -3.1427, 1.2046),
    #     (2.8366, -3.0525, 1.2012),
    # ]:
    original_dtype = G.dtype
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (torch.linalg.norm(X) + eps)  # ensure top singular value <= 1

    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X.to(original_dtype)


zeropower_backends = dict(
    svd=zeropower_via_svd,
    newtonschulz5=zeropower_via_newtonschulz5,
    identity=lambda x, **kwargs: x,
)
