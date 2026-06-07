"""Mamba-2 SSM (Structured State-Space Duality) mixer.

Reference implementation that mirrors HF's
`transformers.models.mamba2.modeling_mamba2.Mamba2Mixer.torch_forward`
(lines 398-585). Prioritizes CORRECTNESS over speed — the chunked SSD
scan is delegated to `api.ops.selective_scan`, a pure-PyTorch reference.

Architecture (all line refs are `modeling_mamba2.py`):

  ┌──────────────────────────────────────────────────────────────────┐
  │ in_proj : Linear(hidden_size -> 2*d_mlp + d_inner + conv_dim     │
  │                                 + num_heads)                     │
  │   (line 166-170; for canonical Mamba-2 d_mlp == 0)               │
  │   slice: [_, _, gate, x_BC, dt]                                  │
  │     gate :  [B, S, d_inner]                                      │
  │     x_BC :  [B, S, conv_dim]   (conv_dim = d_inner + 2*ng*ds)    │
  │     dt   :  [B, S, num_heads]                                    │
  │                                                                  │
  │ x_BC -> transpose(1,2) -> conv1d(depthwise, groups=conv_dim,     │
  │                          padding=conv_kernel-1)                  │
  │       -> [..., :seq_len]   (causal trim, line 351/436)           │
  │       -> SiLU activation (line 351, 354)                         │
  │       -> transpose(1,2)                                          │
  │   slice: [x_for_ssm, B_unrep, C_unrep]                           │
  │     x_for_ssm: [B, S, d_inner]                                   │
  │     B_unrep:   [B, S, ng*ds]    -> view [B, S, ng, ds]           │
  │     C_unrep:   [B, S, ng*ds]    -> view [B, S, ng, ds]           │
  │                                                                  │
  │ dt = softplus(dt + dt_bias)                                      │
  │    .clamp(time_step_limit_low, time_step_limit_high)             │
  │                                                                  │
  │ A = -exp(A_log)        # [num_heads]                             │
  │ A_disc = A * dt        # [B, S, num_heads]                       │
  │ x_disc = x_for_ssm * dt[..., None]   # [B, S, num_heads, head_dim] │
  │                                                                  │
  │ B = repeat_interleave(B_unrep, num_heads // ngroups, dim=2)      │
  │ C = repeat_interleave(C_unrep, num_heads // ngroups, dim=2)      │
  │                                                                  │
  │ y, ssm_state = selective_scan(x_disc, A_disc, B, C, chunk_size)  │
  │                                                                  │
  │ D_residual = D[..., None] * x_for_ssm    # [B, S, num_heads, hd] │
  │ y = y + D_residual                                               │
  │ y = y.reshape(B, S, d_inner)                                     │
  │                                                                  │
  │ y = MambaRMSNormGated(y, gate)   # silu(gate) * fp32-rms         │
  │                                                                  │
  │ out = out_proj(y)                                                │
  └──────────────────────────────────────────────────────────────────┘

State cache (SSMStateCache):
  - conv_state: [B, conv_dim, conv_kernel] — last conv_kernel-1 inputs to conv1d
  - ssm_state:  [B, num_heads, head_dim, d_state] — final per-chunk-end SSM state
"""
from __future__ import annotations
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from api import kvcache as _kvcache, ops, specs, types


class _MambaRMSNormGated(nn.Module):
    """Gated RMSNorm — y = w * rms_normalize(x * silu(gate)).

    Mirrors `MambaRMSNormGated` (modeling_mamba2.py:103-118).

    - x is upcast to fp32 BEFORE the silu-gate multiply.
    - gate is upcast to fp32 INSIDE the function (line 114).
    - rsqrt is computed in fp32.
    - The output is then `w * fp32-result.to(input_dtype)`. The weight
      multiply is in the result's dtype (line 118 `self.weight *
      hidden_states.to(input_dtype)`).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_dtype = x.dtype
        x32 = x.to(torch.float32)
        if gate is not None:
            x32 = x32 * F.silu(gate.to(torch.float32))
        variance = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.eps)
        return self.weight * x32.to(input_dtype)


class Mamba2Mixer(nn.Module):
    """Mamba-2 SSM mixer.

    Source: `modeling_mamba2.py:121-585` (Mamba2Mixer class).
    """

    def __init__(
        self,
        spec: specs.SSDSpec,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.spec = spec
        self.hidden_size = hidden_size
        base = spec.base
        # Cached shape parameters
        self.num_heads = spec.n_heads
        self.head_dim = spec.headdim
        self.n_groups = spec.ngroups
        self.d_state = base.d_state
        self.conv_kernel = base.d_conv
        self.d_inner = base.d_inner
        # Invariant from Mamba2Config.validate_architecture() (config:97-103)
        if self.num_heads * self.head_dim != self.d_inner:
            raise ValueError(
                f"Mamba-2: num_heads*head_dim ({self.num_heads*self.head_dim}) "
                f"must equal d_inner ({self.d_inner})"
            )
        if self.num_heads % self.n_groups != 0:
            raise ValueError(
                f"Mamba-2: num_heads ({self.num_heads}) must be divisible "
                f"by n_groups ({self.n_groups})"
            )
        self.heads_per_group = self.num_heads // self.n_groups
        self.chunk_size = spec.chunk_size
        # conv_dim is the depthwise conv channel count: d_inner + 2 * ng * d_state
        # Source: modeling_mamba2.py:154
        self.conv_dim = self.d_inner + 2 * self.n_groups * self.d_state

        # in_proj projection size: 2*d_mlp + d_inner + conv_dim + num_heads.
        # For canonical Mamba-2 d_mlp = 0 (no extra MLP head); some hybrid
        # families set d_mlp > 0. We expose `d_mlp` via SSDSpec.d_mlp (added
        # as a top-level attr below); for B7 landing we hardcode d_mlp = 0.
        # When extending to families with d_mlp > 0, just add `2*d_mlp` to
        # projection_size and update the split() call accordingly.
        # Source: modeling_mamba2.py:165 (projection_size = intermediate_size +
        # conv_dim + num_heads) and line 410 (d_mlp inferred from total size).
        self.d_mlp = 0
        projection_size = 2 * self.d_mlp + self.d_inner + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            hidden_size, projection_size, bias=base.bias, dtype=dtype,
        )

        # conv1d: depthwise (groups=conv_dim), kernel=conv_kernel, padded
        # conv_kernel-1 on the left to be causal after `[..., :seq_len]`.
        # Source: modeling_mamba2.py:155-162.
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel,
            groups=self.conv_dim,
            padding=self.conv_kernel - 1,
            bias=base.conv_bias,
            dtype=dtype,
        )

        # dt_bias: per-head bias added BEFORE softplus(dt+dt_bias).
        # Source: modeling_mamba2.py:174, used at line 505 (`softplus(dt + dt_bias)`).
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads, dtype=dtype))
        # A_log: per-head log of A_neg parameter. A = -exp(A_log).
        # Source: modeling_mamba2.py:178, used at line 446.
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=dtype))
        # D: per-head skip connection coefficient. D-residual is
        # D[..., None] * hidden_states (line 514).
        self.D = nn.Parameter(torch.ones(self.num_heads, dtype=dtype))

        # Gated RMSNorm before out_proj (line 179).
        self.norm = _MambaRMSNormGated(
            self.d_inner, eps=spec.layer_norm_epsilon, dtype=dtype,
        )

        # out_proj (line 184).
        self.out_proj = nn.Linear(
            self.d_inner, hidden_size, bias=base.bias, dtype=dtype,
        )

        # time_step_limit: clamp range after softplus(dt+dt_bias).
        # Source: modeling_mamba2.py:506 (`torch.clamp(dt, low, high)`).
        self.time_step_limit_low = spec.time_step_limit_low
        self.time_step_limit_high = spec.time_step_limit_high

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[_kvcache.SSMStateCache] = None,
    ) -> torch.Tensor:
        """Mamba-2 mixer forward (prefill path).

        Args:
            hidden_states: [B, S, hidden_size]
            cache: optional SSMStateCache. When provided and
                `cache.has_previous_state == False`, this is a prefill pass:
                we run the chunked SSD scan and write the final ssm_state +
                conv_state back into the cache.
                When `has_previous_state == True` (decode S==1), we run the
                single-step recurrence — NOT YET IMPLEMENTED in B7 (see
                decode TODO note below).

        Returns: [B, S, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. in_proj. Source: modeling_mamba2.py:409.
        projected_states = self.in_proj(hidden_states)
        # Slice. d_mlp is fixed at 0 for canonical Mamba-2 here, so the first
        # two slots are zero-width and the split degenerates to (gate, x_BC, dt).
        # Source: modeling_mamba2.py:411-413.
        _, _, gate, hidden_states_B_C, dt = projected_states.split(
            [self.d_mlp, self.d_mlp, self.d_inner, self.conv_dim, self.num_heads],
            dim=-1,
        )

        # 2. Convolution sequence transformation. Source: modeling_mamba2.py:414-436.
        # Transpose for conv1d which expects [B, C, L].
        x_BC_t = hidden_states_B_C.transpose(1, 2)   # [B, conv_dim, S]
        is_decoding = cache is not None and cache.has_previous_state
        if is_decoding:
            # Single-step decode. NOT IMPLEMENTED in B7 — left as TODO.
            # Source: modeling_mamba2.py:419-427 (`conv_states = update_conv_state(
            # hidden_states_B_C, layer_idx); hidden_states_B_C = torch.sum(
            # conv_states * weight.squeeze(1), -1) + bias; act(...)`).
            raise NotImplementedError(
                "Mamba-2 decode (single-step recurrent state advance) is "
                "deferred — B7 lands prefill numerical gate only."
            )
        else:
            # Prefill: write the LEFT-PADDED final-conv_kernel-slice of the
            # conv input into the cache (line 431-434), if any cache.
            if cache is not None:
                conv_states = F.pad(
                    x_BC_t, (self.conv_kernel - x_BC_t.shape[-1], 0)
                )
                # If S > conv_kernel, the pad amount is negative => no left
                # padding; keep just the last conv_kernel slice.
                if conv_states.shape[-1] != self.conv_kernel:
                    conv_states = conv_states[..., -self.conv_kernel:]
                cache.update_conv_state(conv_states)

            # Apply depthwise conv, slice to seq_len for causality, transpose back.
            conv_out = self.conv1d(x_BC_t)[..., :seq_len]
            # SiLU activation (line 351, 354).
            x_BC_act = F.silu(conv_out).transpose(1, 2)   # [B, S, conv_dim]

        # Slice into x, B, C. Source: modeling_mamba2.py:439-443.
        groups_state = self.n_groups * self.d_state
        x_for_ssm, B_unrep, C_unrep = torch.split(
            x_BC_act, [self.d_inner, groups_state, groups_state], dim=-1,
        )

        # 3. SSM transformation. Source: modeling_mamba2.py:445-577 (else branch).
        A = -torch.exp(self.A_log.float())                # [num_heads]

        # dt: softplus(dt + dt_bias), clamped. Source: lines 505-506.
        # Cast to fp32 for the softplus/clamp; the discretized x and A are kept
        # in fp32 too (HF lines 507-518 do `.float()` on x and `A.to(.dtype)*dt`).
        dt_fp32 = F.softplus(dt + self.dt_bias)
        dt_fp32 = torch.clamp(
            dt_fp32, self.time_step_limit_low, self.time_step_limit_high,
        )

        # x: reshape to [B, S, num_heads, head_dim], cast fp32. Line 507.
        x_disc = x_for_ssm.reshape(batch_size, seq_len, self.num_heads, self.head_dim).float()

        # B, C reshape: [B, S, ng, ds] then repeat_interleave per head.
        # Source: lines 508-511.
        B_ssm = B_unrep.reshape(batch_size, seq_len, self.n_groups, self.d_state).float()
        C_ssm = C_unrep.reshape(batch_size, seq_len, self.n_groups, self.d_state).float()
        B_ssm = B_ssm.repeat_interleave(self.heads_per_group, dim=2,
                                         output_size=self.num_heads)
        C_ssm = C_ssm.repeat_interleave(self.heads_per_group, dim=2,
                                         output_size=self.num_heads)

        # Discretize: x_disc *= dt; A_disc = A * dt. Source: lines 517-518.
        x_disc = x_disc * dt_fp32[..., None]
        A_disc = A.to(dt_fp32.dtype) * dt_fp32      # [B, S, num_heads]

        # Run the SSD scan. selective_scan returns y (truncated to S) +
        # ssm_state. Source: lines 520-567.
        y, ssm_state = ops.selective_scan(
            x_disc, A_disc, B_ssm, C_ssm,
            chunk_size=self.chunk_size,
            initial_state=None,
        )

        # D-residual: per-head D * (unscaled) hidden_states.
        # Source: line 514 (`D_residual = self.D[..., None] *
        # pad_tensor_by_size(hidden_states, pad_size)`). We compute on the
        # already-truncated y so we use x_for_ssm reshaped to num_heads.
        # Note: HF's D_residual is built from the PADDED hidden_states and the
        # add happens after the chunk-reassembled y is reshaped & truncated
        # (line 567-572). Mathematically equivalent post-truncation.
        x_unscaled = x_for_ssm.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim,
        ).float()
        D_residual = self.D[..., None].float() * x_unscaled
        y = y + D_residual

        # Reshape back to [B, S, d_inner]. Source: line 573.
        y = y.reshape(batch_size, seq_len, self.d_inner)

        # Cache final ssm_state, if any. Source: line 576-577.
        if cache is not None:
            cache.update_recurrent_state(ssm_state.to(cache.dtype))

        # Gated RMSNorm with gate. Source: line 579 (`scan_output =
        # self.norm(y, gate)`).
        y = self.norm(y, gate)

        # out_proj. Source: line 584.
        return self.out_proj(y.to(dtype))
