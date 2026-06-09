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


class Mamba1Mixer(nn.Module):
    """Mamba-1 selective-scan mixer (v6 B1).

    Layout (`modeling_mamba.py:58-120` MambaMixer init):
      - in_proj    : Linear(hidden, 2*d_inner, bias=use_bias)
      - conv1d     : depthwise Conv1d(d_inner, kernel=d_conv,
                                      groups=d_inner, padding=d_conv-1,
                                      bias=use_conv_bias)
      - x_proj     : Linear(d_inner, dt_rank + 2*d_state, bias=False)
      - dt_proj    : Linear(dt_rank, d_inner, bias=True)
      - A_log      : Parameter [d_inner, d_state] — A = -exp(A_log)
      - D          : Parameter [d_inner]
      - out_proj   : Linear(d_inner, hidden, bias=use_bias)

    Forward (`modeling_mamba.py::MambaMixer.slow_forward` lines 270-363):

        projected = in_proj(x).transpose(1, 2)             # [B, 2*d_inner, S]
        hidden_states, gate = projected.chunk(2, dim=1)    # each [B, d_inner, S]
        # conv1d + silu
        hidden_states = silu(conv1d(hidden_states)[..., :S])
        # x_proj
        ssm_params = x_proj(hidden_states.transpose(1, 2))  # [B, S, dt_rank+2*ds]
        dt, B, C = split(ssm_params, [dt_rank, ds, ds], dim=-1)
        # Optional dt/B/C layernorms (Jamba — modeling_jamba.py:249-251).
        dt = dt_proj(dt)                                   # [B, S, d_inner]
        dt = softplus(dt).transpose(1, 2)                  # [B, d_inner, S]
        # SSM scan (api.ops.selective_scan_mamba1).
        y, ssm_state = selective_scan_mamba1(
            hidden_states, dt, A=-exp(A_log), B, C, D, gate,
        )
        out = out_proj(y.transpose(1, 2))                  # [B, S, hidden]
    """

    def __init__(
        self,
        spec: specs.SSMSpec,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if spec.kind != types.SSMKind.MAMBA1:
            raise ValueError(
                f"Mamba1Mixer requires spec.kind=MAMBA1, got {spec.kind}"
            )
        if spec.dt_rank is None:
            raise ValueError("Mamba1Mixer requires spec.dt_rank")
        self.spec = spec
        self.hidden_size = hidden_size
        self.d_inner = spec.d_inner
        self.d_state = spec.d_state
        self.d_conv = spec.d_conv
        self.dt_rank = spec.dt_rank

        # in_proj: hidden -> 2 * d_inner (chunk(2) → hidden, gate).
        self.in_proj = nn.Linear(
            hidden_size, 2 * self.d_inner, bias=spec.bias, dtype=dtype,
        )

        # Depthwise conv1d.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=spec.conv_bias,
            dtype=dtype,
        )

        # x_proj: d_inner -> dt_rank + 2*d_state.
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.d_state, bias=False, dtype=dtype,
        )

        # dt_proj: dt_rank -> d_inner, with bias.
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, dtype=dtype,
        )

        # A_log: [d_inner, d_state]; D: [d_inner].
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, self.d_state, dtype=dtype))
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=dtype))

        # out_proj: d_inner -> hidden.
        self.out_proj = nn.Linear(
            self.d_inner, hidden_size, bias=spec.bias, dtype=dtype,
        )

        # v6 B1: optional dt/B/C layernorms (Jamba).
        if spec.dt_layernorm is not None:
            from api import norm as _norm
            self.dt_layernorm = _norm.RMSNorm(spec.dt_layernorm, self.dt_rank, dtype=dtype)
        else:
            self.dt_layernorm = None
        if spec.b_layernorm is not None:
            from api import norm as _norm
            self.b_layernorm = _norm.RMSNorm(spec.b_layernorm, self.d_state, dtype=dtype)
        else:
            self.b_layernorm = None
        if spec.c_layernorm is not None:
            from api import norm as _norm
            self.c_layernorm = _norm.RMSNorm(spec.c_layernorm, self.d_state, dtype=dtype)
        else:
            self.c_layernorm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[_kvcache.SSMStateCache] = None,
    ) -> torch.Tensor:
        """Prefill forward. cache=None or has_previous_state=False.

        Args:
            hidden_states: [B, S, hidden_size]
            cache: optional SSMStateCache (not used for decode in v6 B1).

        Returns: [B, S, hidden_size]
        """
        B_n, S, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. in_proj → chunk to (hidden_states, gate).
        # Source: modeling_mamba.py:274-275.
        projected = self.in_proj(hidden_states).transpose(1, 2)   # [B, 2*d_inner, S]
        x, gate = projected.chunk(2, dim=1)                       # each [B, d_inner, S]

        # 2. Depthwise causal conv1d + silu.
        # Source: modeling_mamba.py:289-306 (slow_forward, no cache path).
        if cache is not None and cache.has_previous_state:
            raise NotImplementedError(
                "Mamba-1 decode (single-step recurrent state advance) is "
                "deferred — v6 B1 lands prefill numerical gate only."
            )
        if cache is not None:
            conv_state = F.pad(x, (self.d_conv - x.shape[-1], 0))
            if conv_state.shape[-1] != self.d_conv:
                conv_state = conv_state[..., -self.d_conv:]
            cache.update_conv_state(conv_state)
        x = self.conv1d(x)[..., :S]
        # Use SiLU activation (Mamba-1 hidden_act='silu').
        x = F.silu(x)

        # 3. x_proj → split dt, B, C.
        # Source: modeling_mamba.py:313-316.
        ssm_params = self.x_proj(x.transpose(1, 2))               # [B, S, dt_rank+2*ds]
        dt, B_in, C_in = torch.split(
            ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1,
        )

        # 3b. Jamba-style intra-mixer norms on dt/B/C (modeling_jamba.py:324-326).
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.b_layernorm is not None:
            B_in = self.b_layernorm(B_in)
        if self.c_layernorm is not None:
            C_in = self.c_layernorm(C_in)

        # 4. dt_proj + softplus.
        # Source: modeling_mamba.py:317-318.
        discrete_dt = self.dt_proj(dt)                            # [B, S, d_inner]
        discrete_dt = F.softplus(discrete_dt).transpose(1, 2)     # [B, d_inner, S]

        # 5. SSM scan.
        A = -torch.exp(self.A_log.float())                        # [d_inner, d_state]
        scan_output, ssm_state = ops.selective_scan_mamba1(
            x, discrete_dt, A, B_in, C_in, self.D, gate,
            initial_state=None,
        )

        if cache is not None:
            cache.update_recurrent_state(ssm_state.to(cache.dtype))

        # 6. out_proj.
        # scan_output: [B, d_inner, S] → transpose → out_proj → [B, S, hidden]
        # Source: modeling_mamba.py:362.
        out = self.out_proj(scan_output.transpose(1, 2).to(dtype))
        return out


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


# ---------------------------------------------------------------------------
# v7 P3: Qwen3-Next Gated DeltaNet linear-attention mixer
# ---------------------------------------------------------------------------


class _Qwen3NextRMSNormGated(nn.Module):
    """v7 P3: Qwen3-Next gated RMSNorm — silu(gate) * rms_normalize(x).

    Differs from `_MambaRMSNormGated` (Mamba-2) in the multiply order:
    Qwen3-Next normalises x first (with learned weight applied AFTER cast back
    to input dtype), THEN multiplies by silu(gate) in fp32, then casts.

    Source: `transformers/models/qwen3_next/modeling_qwen3_next.py:67-82`.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        h = self.weight * h.to(input_dtype)
        h = h * F.silu(gate.to(torch.float32))
        return h.to(input_dtype)


class GatedDeltaNetMixer(nn.Module):
    """v7 P3: Qwen3-Next Gated DeltaNet linear-attention mixer.

    Module layout (matches HF state-dict keys for direct load):
        - in_proj_qkvz   : Linear(hidden, 2*key_dim + 2*value_dim, bias=False)
        - in_proj_ba     : Linear(hidden, 2*num_v_heads, bias=False)
        - conv1d         : Conv1d(conv_dim, conv_dim, kernel=conv_kernel,
                                  groups=conv_dim, padding=conv_kernel-1,
                                  bias=False) where conv_dim = 2*key_dim + value_dim
        - A_log          : Parameter [num_v_heads]
        - dt_bias        : Parameter [num_v_heads]
        - norm           : Qwen3NextRMSNormGated(head_v_dim, eps)
        - out_proj       : Linear(value_dim, hidden, bias=False)

    Forward (mirrors `Qwen3NextGatedDeltaNet.forward`,
    `modeling_qwen3_next.py:595-717`):

        projected_qkvz = in_proj_qkvz(x)
        projected_ba   = in_proj_ba(x)
        q, k, v, z, b, a = fix_query_key_value_ordering(...)
        # depthwise causal conv on cat(q, k, v).transpose(1,2)
        mixed = silu(conv1d(cat(q, k, v).T)[..., :S]).T
        q, k, v = split(mixed, [key_dim, key_dim, value_dim], dim=-1).view(...)
        beta = sigmoid(b)
        g = -exp(A_log.float()) * softplus(a.float() + dt_bias)
        # GQA repeat for q/k.
        if n_v_heads > n_k_heads:
            q = repeat_interleave(q, n_v_heads // n_k_heads, dim=2)
            k = repeat_interleave(k, n_v_heads // n_k_heads, dim=2)
        out, _ = gated_delta_step(q, k, v, g, beta, ...,
                                  use_qk_l2norm=True)
        out = norm(out, z)             # silu-gated RMSNorm
        out = out_proj(out.reshape(B, S, value_dim))

    Sources cited inline below.
    """

    def __init__(
        self,
        spec: specs.GatedDeltaNetSpec,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if spec.num_v_heads % spec.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({spec.num_v_heads}) must be divisible by "
                f"num_k_heads ({spec.num_k_heads})"
            )
        self.spec = spec
        self.hidden_size = hidden_size
        self.num_v_heads = spec.num_v_heads
        self.num_k_heads = spec.num_k_heads
        self.head_k_dim = spec.head_k_dim
        self.head_v_dim = spec.head_v_dim
        self.key_dim = spec.head_k_dim * spec.num_k_heads
        self.value_dim = spec.head_v_dim * spec.num_v_heads
        self.conv_kernel_size = spec.conv_kernel

        # Conv front-end over (q | k | v). Source: modeling_qwen3_next.py:517-525.
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            dtype=dtype,
        )

        # Input projections. Source: modeling_qwen3_next.py:528-531.
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(hidden_size, projection_size_qkvz,
                                      bias=False, dtype=dtype)
        self.in_proj_ba = nn.Linear(hidden_size, projection_size_ba,
                                    bias=False, dtype=dtype)

        # Time-step bias + per-head log-decay. Source: modeling_qwen3_next.py:535-538.
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, dtype=dtype))
        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(A.log().to(dtype))

        # Qwen3NextRMSNormGated on the per-head output. Source: line 540-549.
        self.norm = _Qwen3NextRMSNormGated(
            self.head_v_dim, eps=spec.norm_eps, dtype=dtype,
        )

        # Output projection.
        self.out_proj = nn.Linear(self.value_dim, hidden_size,
                                  bias=False, dtype=dtype)

    def _fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Replicates `Qwen3NextGatedDeltaNet.fix_query_key_value_ordering`
        (modeling_qwen3_next.py:566-593) — splits and reshapes the fused QKVZ
        and BA projections into (q, k, v, z, b, a)."""
        B, S, _ = mixed_qkvz.shape
        H_v_per_k = self.num_v_heads // self.num_k_heads
        # Per group-of-k-heads: head_k_dim (q) + head_k_dim (k) +
        # H_v_per_k * head_v_dim (v) + H_v_per_k * head_v_dim (z).
        shape_qkvz = (B, S, self.num_k_heads,
                      2 * self.head_k_dim + 2 * self.head_v_dim * H_v_per_k)
        shape_ba = (B, S, self.num_k_heads, 2 * H_v_per_k)
        mqkvz = mixed_qkvz.view(*shape_qkvz)
        mba = mixed_ba.view(*shape_ba)
        split_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            H_v_per_k * self.head_v_dim,
            H_v_per_k * self.head_v_dim,
        ]
        split_ba = [H_v_per_k, H_v_per_k]
        q, k, v, z = torch.split(mqkvz, split_qkvz, dim=3)
        b, a = torch.split(mba, split_ba, dim=3)
        # Reshape v, z to [B, S, num_v_heads, head_v_dim].
        v = v.reshape(B, S, -1, self.head_v_dim)
        z = z.reshape(B, S, -1, self.head_v_dim)
        b = b.reshape(B, S, self.num_v_heads)
        a = a.reshape(B, S, self.num_v_heads)
        return q, k, v, z, b, a

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[object] = None,
    ) -> torch.Tensor:
        """Forward.

        Args:
            x: [B, S, hidden_size]
            cache: reserved — Qwen3-Next caches (conv_state, recurrent_state)
                per layer. Not implemented in the v7 P3 landing; we run the
                full sequence in-line each call.
        """
        B, S, _ = x.shape

        # Projections. Source: lines 618-620.
        projected_qkvz = self.in_proj_qkvz(x)
        projected_ba = self.in_proj_ba(x)
        q, k, v, z, b, a = self._fix_query_key_value_ordering(
            projected_qkvz, projected_ba,
        )
        # Flatten per-token to [B, S, key_dim/value_dim] for the conv.
        # Source: line 621.
        q = q.reshape(B, S, -1)
        k = k.reshape(B, S, -1)
        v_flat = v.reshape(B, S, -1)

        # Depthwise causal conv on cat(q, k, v).transpose(1, 2). Source:
        # lines 623-624, 654 (fallback path).
        mixed_qkv = torch.cat((q, k, v_flat), dim=-1).transpose(1, 2)
        mixed_qkv = self.conv1d(mixed_qkv)[:, :, : S]
        mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)

        # Re-split + reshape into per-head tensors. Source: lines 658-670.
        q_dim = self.key_dim
        v_dim = self.value_dim
        q, k, v_post = torch.split(mixed_qkv, [q_dim, q_dim, v_dim], dim=-1)
        q = q.reshape(B, S, -1, self.head_k_dim)
        k = k.reshape(B, S, -1, self.head_k_dim)
        v_post = v_post.reshape(B, S, -1, self.head_v_dim)

        # beta + g. Source: lines 672-674.
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(
            a.float() + self.dt_bias.float()
        )

        # GQA repeat-interleave for q/k. Source: lines 675-677.
        H_v_per_k = self.num_v_heads // self.num_k_heads
        if H_v_per_k > 1:
            q = q.repeat_interleave(H_v_per_k, dim=2)
            k = k.repeat_interleave(H_v_per_k, dim=2)

        # Sequential reference recurrence. The HF fast path uses
        # chunk_gated_delta_rule; for the v7 P3 landing we use the per-step
        # form because it makes numerical comparison straightforward.
        # Source for kernel call: lines 680-702.
        out, _ = ops.gated_delta_step(
            q, k, v_post, g=g, beta=beta,
            initial_state=None, use_qk_l2norm=True,
            eps=1e-6,
        )

        # Gated RMSNorm + reshape + out_proj. Source: lines 708-716.
        # HF reshapes `core_attn_out`/`z` to 2D before norm. We do the same.
        z_shape = z.shape
        out_flat = out.reshape(-1, out.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        out_normed = self.norm(out_flat, z_flat).reshape(z_shape)
        out_normed = out_normed.reshape(B, S, -1)
        return self.out_proj(out_normed)
