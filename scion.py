import torch
import torch.distributed.tensor

__all__ = [
    "Scion",
]


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


class Scion(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr,
        momentum,
        norm_factor="spectral",
        is_unconstrained=False,
        backend="newtonschulz5",
        backend_steps=5,
        is_light=False,
        nesterov=False,
        eps=1e-20,
    ):

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            eps=eps,
            norm_factor=norm_factor,
            backend=backend,
            backend_steps=backend_steps,
        )
        self.is_light = is_light
        # NB: use default momentum here, param groups can have its own
        #     values
        self.use_momentum = momentum > 0 and momentum < 1
        self.is_unconstrained = is_unconstrained
        print(
            f"Scion optimizer (is_light={self.is_light}, is_unconstrained={self.is_unconstrained})"
        )
        super().__init__(params, defaults)
        if self.is_light:
            # Initialize state
            self._store_grads_in_state()
            # Do not pass `self` through syntactic sugar. We need the
            # argument to not be populated.
            self.register_state_dict_pre_hook(
                type(self)._store_grads_in_state,
            )
            self.register_load_state_dict_post_hook(
                type(self)._load_grads_from_state,
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            nesterov = group["nesterov"]
            momentum = group["momentum"]
            param_kwargs = {
                "eps": group["eps"],
                "norm_factor": group["norm_factor"],
                "zeropower_backend": zeropower_backends[group["backend"]],
                "backend_steps": group["backend_steps"],
            }
            # NB: here assume that normalisation with norm_factor is
            #     done across non-sharded axis, so can skip
            #     communication
            if self.is_light and nesterov:
                raise NotImplementedError(
                    "Nesterov momentum is not supported for light mode. "
                    "Please set nesterov=False."
                )

            for p in group["params"]:
                g = self.get_momentum_or_grad(
                    p,
                    momentum,
                    nesterov,
                    update_buffer=True,
                )
                if g is None:
                    continue
                update = self.lmo(g, **param_kwargs)

                if update.shape != p.data.shape:
                    raise RuntimeError(
                        f"Shape mismatch: g.shape={g.shape}, p.data.shape={p.data.shape}"
                    )

                if not self.is_unconstrained:
                    p.data.mul_(1 - lr)
                p.data.add_(update, alpha=-lr)

                if self.is_light and self.use_momentum:
                    p.grad.mul_(1 - momentum)

        return loss

    @torch.no_grad()
    def lmo(self, g, eps, norm_factor, zeropower_backend, backend_steps):
        # NB: make sure this function does not modify the grad inplace
        #     since it is also called during the log of gradients
        if g.ndim == 2:
            g = zeropower_backend(g, steps=backend_steps, eps=eps)
            g = self.normalise_grad(g, norm_factor=norm_factor, eps=eps)
            return g

        else:
            raise ValueError(
                f"Unsupported tensor shape: {g.shape}. " "Expected 2D or 3D tensor."
            )

    @torch.no_grad()
    def normalise_grad(self, g, norm_factor, eps):
        if norm_factor == "spectral":
            g = g * (g.size(0) / g.size(1)) ** 0.5
        elif norm_factor == "image_spectral":
            # Image domain norm as described in Scion paper.
            g = g * max((g.size(0) / g.size(1)) ** 0.5, 1)
        elif norm_factor.startswith("embed"):
            # NB: here assume shape [vocab_size, embed_dim]
            rms_values = torch.sqrt(g.pow(2).sum(axis=1, keepdim=True))
            g = g / (rms_values + eps)
            if norm_factor == "embed_linear":
                g = g * g.size(1)
            elif norm_factor == "embed_sqrt":
                g = g * g.size(1) ** 0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")
        elif norm_factor.startswith("unembed"):
            rms_values = torch.sqrt(g.pow(2).sum(axis=1, keepdim=True))
            g = g / (rms_values + eps)
            if norm_factor == "unembed_linear":
                g = g / g.size(1)
            elif norm_factor == "unembed_sqrt":
                g = g / g.size(1) ** 0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")
        elif norm_factor == "none":
            pass
        else:
            raise ValueError(f"Unknown norm_factor: {norm_factor}")

        return g

    @torch.no_grad()
    def get_momentum_or_grad(self, p, momentum, nesterov, update_buffer=True):
        g = p.grad
        if g is None or not p.requires_grad:
            return None

        if not self.is_light and self.use_momentum:
            state = self.state[p]
            if "momentum_buffer" not in state.keys():
                if update_buffer:
                    state["momentum_buffer"] = torch.zeros_like(g)
                else:
                    raise ValueError(
                        "Momentum buffer not found in optimizer state. "
                        "Please check if the optimizer is initialized correctly."
                    )
            buf = state["momentum_buffer"]
            if update_buffer:
                buf.mul_(1 - momentum).add_(g, alpha=momentum)
            else:
                buf = buf.mul(1 - momentum).add(g, alpha=momentum)
            g = buf if not nesterov else buf.mul(1 - momentum).add(g, alpha=momentum)

        return g

    def __getstate__(self):
        self._store_grads_in_state()
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        self._load_grads_from_state()

    def _store_grads_in_state(self):
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, torch.Tensor) and param.grad is not None:
                    self.state.setdefault(param, {})["grad_state"] = param.grad

    def _load_grads_from_state(self):
        for param, state in self.state.items():
            if "grad_state" in state:
                param.grad = state["grad_state"]
            elif isinstance(param, torch.Tensor):
                param.grad = None
