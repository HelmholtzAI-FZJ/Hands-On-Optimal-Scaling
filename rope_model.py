"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

"""
Scion-change
1. Use no-parameter RMSNorm instead of LayerNorm
2. Use rotary position embedding | rather than learned position embedding [larnble PE is not well defined]
3. No weight tying (lm_head and token embedding are not tied)
4. Use orthogonal initialization for hidden weights
5. Use Col/Row-wise init for token embedding and lm_head
"""
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # SCION_CHANGE: Split QKV following https://github.com/KellerJordan/modded-nanogpt/
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.mup_enabled = config.mup_enabled
        self.mup_disable_attention_scaling = config.mup_disable_attention_scaling
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(
            k, (k.size(-1),)
        )  # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.mup_enabled and not self.mup_disable_attention_scaling:
            ### Begin muP code ###
            attention_scale = 1.0 / q.size(-1)
            ### End muP code ###
        else:
            attention_scale = 1.0 / math.sqrt(q.size(-1))

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=attention_scale
        )
        y = (
            y.transpose(1, 2).contiguous().view_as(x)
        )  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        if config.no_parameters_rmsnorm:
            ln_build = partial(nn.RMSNorm, eps=1e-6, elementwise_affine=False)
        else:
            ln_build = partial(LayerNorm, bias=config.bias)

        self.ln_1 = ln_build(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = ln_build(config.n_embd)
        self.mlp = MLP(config)
        self.residual_scaling = (
            1 / (config.depth_multiplier**config.depth_alpha_exp)
            if config.depth_alpha_enabled
            else 1.0
        )

    def forward(self, x):
        ### Begin CompleteP code ###
        x = x + self.residual_scaling * self.attn(self.ln_1(x))
        x = x + self.residual_scaling * self.mlp(self.ln_2(x))
        ### End CompleteP code ###
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
    init_std: float = 0.02
    mup_enabled: bool = (
        False  # Whether to use muP. If False then all other mup variables are ignored
    )
    mup_disable_attention_scaling: bool = False  # Disables mup attention scaling
    mup_disable_hidden_lr_scaling: bool = False  # Disables mup hidden LR scaling
    mup_width_multiplier: float = (
        1  # `mup_width_multiplier = width / base_width` where base_width is typically 256
    )
    mup_input_alpha: float = (
        1  # Optional tunable multiplier applied to input embedding forward pass output
    )
    mup_output_alpha: float = (
        1  # Optional tunable multiplier applied to output unembedding forward pass output
    )
    depth_alpha_enabled: bool = False
    depth_multiplier: float = 1.0  # depth_multiplier = depth / base_depth`
    depth_alpha_exp: float = (
        1.0  # a float in the range [0.5, 1] that controls how residual branches are scaled as a function of depth. This results in residual connections of the type x = x + depth_multiplier**(-depth_alpha_exp) * branch(x) with LR correction eta *= depth_multiplier**(depth_alpha_exp-1)
    )
    no_parameters_rmsnorm: bool = False  # Whether to use RMSNorm instead of LayerNorm


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        if config.no_parameters_rmsnorm:
            ln_build = partial(nn.RMSNorm, eps=1e-6, elementwise_affine=False)
        else:
            ln_build = partial(LayerNorm, bias=config.bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=ln_build(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = (
        #     self.lm_head.weight
        # )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if config.mup_enabled:
                ### Begin muP code ###
                # Adjust hidden weight initialization variance by 1 / mup_width_multiplier
                if pn.endswith("c_attn.weight") or pn.endswith("c_fc.weight"):
                    torch.nn.init.normal_(
                        p,
                        mean=0.0,
                        std=config.init_std / math.sqrt(config.mup_width_multiplier),
                    )
                elif pn.endswith("c_proj.weight"):
                    torch.nn.init.normal_(
                        p,
                        mean=0.0,
                        std=config.init_std / math.sqrt(config.mup_width_multiplier),
                    )
                ### End muP code ###
            elif pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        # pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        # x = self.transformer.drop((tok_emb + pos_emb))
        x = self.transformer.drop(tok_emb)
        if self.config.mup_enabled:
            ### Begin muP code ###
            x *= self.config.mup_input_alpha
            ### End muP code ###
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            if self.config.mup_enabled:
                ### Begin muP code ###
                # Scaling `x` instead of `logits` allows coord check to log change
                x *= self.config.mup_output_alpha / self.config.mup_width_multiplier
                ### End muP code ###
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self, weight_decay, learning_rate, betas, adam_eps, device_type
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        if self.config.mup_enabled and not self.config.mup_disable_hidden_lr_scaling:
            emb_params = []
            hidden_ln_params = []
            hidden_weight_params = []
            hidden_bias_params = []
            final_ln_params = []
            for n, p in param_dict.items():
                if n in ("transformer.wte.weight", "transformer.wpe.weight"):
                    emb_params.append(p)
                elif ".ln_" in n and not ".ln_f." in n:
                    hidden_ln_params.append(p)
                elif (
                    n.endswith("c_attn.weight")
                    or n.endswith("c_fc.weight")
                    or n.endswith("c_proj.weight")
                ):
                    hidden_weight_params.append(p)
                elif (
                    n.endswith("c_attn.bias")
                    or n.endswith("c_fc.bias")
                    or n.endswith("c_proj.bias")
                ):
                    hidden_bias_params.append(p)
                elif ".ln_f." in n:
                    final_ln_params.append(p)
                else:
                    raise Exception(f"Unhandled parameter {n}")
            depth_lr_scaling = self.config.depth_multiplier ** (
                self.config.depth_alpha_exp - 1
            )
            width_lr_scaling = 1 / self.config.mup_width_multiplier
            if self.config.depth_alpha_enabled:
                ### Begin CompleteP code ###
                adam_eps *= (1 / self.config.mup_width_multiplier) * (
                    self.config.depth_multiplier ** (-1 * self.config.depth_alpha_exp)
                )
                optim_groups = [
                    {
                        "params": emb_params,
                        "weight_decay": weight_decay,
                        "lr_scale": 1.0,
                    },
                    {
                        "params": hidden_ln_params,
                        "weight_decay": 0.0,
                        "lr_scale": depth_lr_scaling,
                    },
                    {
                        "params": hidden_weight_params,
                        "weight_decay": weight_decay / width_lr_scaling,
                        "lr_scale": width_lr_scaling * depth_lr_scaling,
                    },
                    {
                        "params": hidden_bias_params,
                        "weight_decay": 0.0,
                        "lr_scale": depth_lr_scaling,
                    },
                    {
                        "params": final_ln_params,
                        "weight_decay": 0.0,
                        "lr_scale": 1.0,
                    },
                ]
                ### End CompleteP code ###
            else:
                ### Begin muP code ###
                optim_groups = [
                    {
                        "params": emb_params,
                        "weight_decay": weight_decay,
                        "lr_scale": 1.0,
                    },
                    {
                        "params": hidden_ln_params,
                        "weight_decay": 0.0,
                        "lr_scale": 1.0,
                    },
                    {
                        "params": hidden_weight_params,
                        "weight_decay": weight_decay,
                        "lr_scale": width_lr_scaling,
                    },
                    {
                        "params": hidden_bias_params,
                        "weight_decay": 0.0,
                        "lr_scale": 1.0,
                    },
                    {
                        "params": final_ln_params,
                        "weight_decay": 0.0,
                        "lr_scale": 1.0,
                    },
                ]
                ### End muP code ###
        else:
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, eps=adam_eps, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
