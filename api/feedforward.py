"""Feed-forward / channel-mixer building blocks.

M1 supports SwiGLU only — the dominant SLM channel mixer (Llama, Qwen, Mistral).
B0.5 adds GeGLU (Gemma) using the gelu_pytorch_tanh activation.
B2a adds fused gate_up (Phi-3).

B5 adds MoE: routed experts with optional shared experts and one of two
router families:

- "softmax" (DeepSeek-V2): F.linear(x_fp32, weight_fp32) → softmax → top-k
  (optionally group-limited greedy via per-group max). Source:
  modeling_deepseek_v2.py:85-130.
- "sigmoid_plus_bias" (DeepSeek-V3): F.linear(x_fp32, weight_fp32) →
  sigmoid → add(score_correction_bias) for routing CHOICE only, but the
  selected WEIGHTS are gathered from the bias-free sigmoid output, then
  optionally norm_topk_prob, then scale. Source:
  modeling_deepseek_v3.py:139-247.

Expert layout follows HF DeepseekV2Experts (modeling_deepseek_v2.py:46-82)
and V3 NaiveMoe (modeling_deepseek_v3.py:154-191):
- gate_up_proj: [n_experts, 2*moe_intermediate_size, hidden_size]
  (combined SwiGLU gate+up — chunk(2, dim=-1) at runtime).
- down_proj:    [n_experts, hidden_size, moe_intermediate_size].
"""
from __future__ import annotations
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from api import ops, specs, types


class FeedForward(nn.Module):
    def __init__(self, spec: specs.FFNSpec, hidden_size: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        # v5-phase2 V3: SWIGLU now accepts RELU2 (BitNet b1.58 uses
        # `gate_kind=SWIGLU + activation=RELU2`). Source:
        # `transformers/models/bitnet/modeling_bitnet.py:70-77`
        # (BitNetMLP with gate_proj/up_proj/down_proj + act_fn applied
        # to the gate path).
        if spec.gate_kind not in (
            types.GateKind.SWIGLU,
            types.GateKind.GEGLU,
            types.GateKind.GELU_ONLY,
        ):
            raise NotImplementedError(
                f"v6 A1: SWIGLU, GEGLU, and GELU_ONLY only, got {spec.gate_kind}"
            )
        if spec.gate_kind == types.GateKind.SWIGLU and spec.activation not in (
            types.Activation.SILU, types.Activation.RELU2,
        ):
            raise NotImplementedError(
                f"v5-phase2: SWIGLU requires SILU or RELU2, got {spec.activation}"
            )
        if spec.gate_kind == types.GateKind.GEGLU and spec.activation != types.Activation.GELU:
            raise NotImplementedError(
                f"B0.5: GEGLU requires GELU activation, got {spec.activation}"
            )
        if spec.gate_kind == types.GateKind.GELU_ONLY and spec.activation != types.Activation.GELU_EXACT:
            # v6 A1: MPT (modeling_mpt.py:143 `nn.GELU(approximate='none')`)
            # and Falcon-7B (configuration_falcon.py:88 `activation='gelu'`,
            # routed via get_activation('gelu') → nn.GELU(approximate='none'))
            # both use the EXACT (erf-based) GELU, not the tanh approximation.
            raise NotImplementedError(
                f"v6 A1: GELU_ONLY requires GELU_EXACT activation, got {spec.activation}"
            )
        if spec.gate_kind == types.GateKind.GELU_ONLY and spec.fused_gate_up:
            raise NotImplementedError(
                "v6 A1: GELU_ONLY is an UNGATED FFN — fused_gate_up has no meaning"
            )
        self.spec = spec
        self.hidden_size = hidden_size
        I = spec.intermediate_size
        # v6 A1: GELU_ONLY (MPT, Falcon-7B). Ungated FFN: up_proj -> exact GELU
        # -> down_proj. No gate_proj. Source:
        # - modeling_mpt.py:137-155 (MptMLP — `up_proj`, `GELU(approximate='none')`,
        #   `down_proj`)
        # - modeling_falcon.py:531-544 (FalconMLP — `dense_h_to_4h`,
        #   `act = get_activation('gelu')`, `dense_4h_to_h`)
        # The HF Falcon FalconMLP uses the same dense_h_to_4h/dense_4h_to_h
        # names but the math is identical to MPT's up_proj/down_proj.
        if spec.gate_kind == types.GateKind.GELU_ONLY:
            self.gate_up_proj = None
            self.gate_proj = None
            self.up_proj = nn.Linear(hidden_size, I, bias=spec.up_bias, dtype=dtype)
            self.down_proj = nn.Linear(I, hidden_size, bias=spec.down_bias, dtype=dtype)
            if spec.ffn_sub_norm is not None:
                raise NotImplementedError(
                    "v6 A1: GELU_ONLY + ffn_sub_norm is not in BitNet's mixing — "
                    "no upstream pairing observed."
                )
            self.ffn_sub_norm = None
            return
        if spec.fused_gate_up:
            # B2a: Phi-3 fused gate/up — one big projection of size 2*I,
            # chunked at forward time. Source: modeling_phi3.py:54 (
            # `gate_up_proj = Linear(hidden_size, 2 * intermediate_size, bias=False)`)
            # and 58-62 (`up_states.chunk(2, dim=-1)` → `up_states * act(gate)`).
            if spec.gate_bias != spec.up_bias:
                raise NotImplementedError(
                    "fused_gate_up requires gate_bias == up_bias (Phi-3: both False)"
                )
            self.gate_up_proj = nn.Linear(hidden_size, 2 * I, bias=spec.gate_bias, dtype=dtype)
            self.gate_proj = None
            self.up_proj = None
        else:
            self.gate_up_proj = None
            self.gate_proj = nn.Linear(hidden_size, I, bias=spec.gate_bias, dtype=dtype)
            self.up_proj = nn.Linear(hidden_size, I, bias=spec.up_bias, dtype=dtype)
        self.down_proj = nn.Linear(I, hidden_size, bias=spec.down_bias, dtype=dtype)

        # v5-phase2 V3: BitNet FFN sub-norm — RMSNorm on the gated
        # activation BEFORE down_proj. Source: modeling_bitnet.py:74, 77.
        if spec.ffn_sub_norm is not None:
            if spec.ffn_sub_norm.kind != types.NormKind.RMS:
                raise ValueError("ffn_sub_norm: RMS kind only")
            from api import norm as _norm  # local to avoid cycle
            self.ffn_sub_norm = _norm.RMSNorm(spec.ffn_sub_norm, I, dtype=dtype)
        else:
            self.ffn_sub_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # v6 A1: GELU_ONLY ungated path — MPT/Falcon-7B.
        if self.spec.gate_kind == types.GateKind.GELU_ONLY:
            # exact GELU = F.gelu(x, approximate='none') — see ops.gelu_exact.
            return self.down_proj(ops.gelu_exact(self.up_proj(x)))
        if self.spec.fused_gate_up:
            # Phi-3 order: chunk(2, dim=-1) → (gate, up) — verify modeling_phi3.py:61.
            up_states = self.gate_up_proj(x)
            gate, up = up_states.chunk(2, dim=-1)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        if self.spec.gate_kind == types.GateKind.SWIGLU:
            # v5-phase2 V3: activation may be SILU (Llama/Qwen) or RELU2
            # (BitNet b1.58, Nemotron 3 squared-ReLU).
            if self.spec.activation == types.Activation.RELU2:
                act = ops.relu2(gate)
            else:
                act = ops.silu(gate)
        else:  # GEGLU — use gelu_pytorch_tanh per Gemma's signature
            act = ops.gelu_pytorch_tanh(gate)
        inner = ops.mul(act, up)
        # v5-phase2 V3: BitNet ffn_sub_norm — RMSNorm on the gated
        # product BEFORE down_proj. Source: modeling_bitnet.py:77.
        if self.ffn_sub_norm is not None:
            inner = self.ffn_sub_norm(inner)
        return self.down_proj(inner)


class MoE(nn.Module):
    """Mixture-of-Experts channel mixer (DeepSeek-V2 / V3 family).

    Reference: modeling_deepseek_v2.py::DeepseekV2Moe (L85-130)
    and modeling_deepseek_v3.py::DeepseekV3MoE (L194-247).

    Module layout (matches HF state dict naming for direct load):
    - gate: nn.Linear(hidden, n_experts, bias=False) for V2 router
            ("softmax"). For V3 ("sigmoid_plus_bias") we expose a wrapper
            with a `weight` parameter and a non-learnable buffer
            `e_score_correction_bias` (shape [n_experts], dtype fp32 even
            when the model is fp16/bf16 — HF marks it
            `_keep_in_fp32_modules_strict`).
    - experts.gate_up_proj: [n_experts, 2 * I, hidden]
    - experts.down_proj:    [n_experts, hidden, I]
    - shared_experts: an FFN block of intermediate_size = I * n_shared_experts
      (None when n_shared_experts == 0).

    Forward (dense scatter — simple, correct, deferred grouped-GEMM).
    """

    def __init__(
        self,
        spec: specs.MoESpec,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if spec.router_kind not in (
            "softmax",
            "sigmoid_plus_bias",
            "topk_then_softmax_with_bias",
            "hash",
        ):
            raise NotImplementedError(
                f"B5/v6 A3/v7 P1: router_kind must be 'softmax', "
                f"'sigmoid_plus_bias', 'topk_then_softmax_with_bias', or 'hash', "
                f"got {spec.router_kind!r}"
            )
        if spec.router_kind == "hash":
            # v7 P1: hash routing requires a vocab size for the `tid2eid`
            # buffer. Group routing is incompatible with hash (selection is
            # deterministic, not score-driven). Source:
            # modeling_deepseek_v4.py:1067.
            if spec.hash_vocab_size is None or spec.hash_vocab_size <= 0:
                raise ValueError(
                    "router_kind='hash' requires hash_vocab_size > 0 "
                    "(size of the tid2eid token-id → expert-id table)"
                )
            if spec.group_routing is not None:
                raise NotImplementedError(
                    "router_kind='hash' is incompatible with group_routing "
                    "(hash selection is deterministic)"
                )
            if spec.hash_score_fn != "sigmoid":
                raise NotImplementedError(
                    f"router_kind='hash' only supports score_fn='sigmoid' "
                    f"(DeepSeek-V4 default), got {spec.hash_score_fn!r}"
                )
        if spec.expert_ffn is None:
            raise ValueError("MoESpec.expert_ffn is required")
        ffn = spec.expert_ffn
        # v6 A3: GPT-OSS clamped SwiGLU is the only expert_kind != 'swiglu'.
        if spec.expert_kind not in ("swiglu", "gpt_oss_clamped_swiglu"):
            raise NotImplementedError(
                f"v6 A3: expert_kind must be 'swiglu' or 'gpt_oss_clamped_swiglu', "
                f"got {spec.expert_kind!r}"
            )
        if spec.expert_kind == "swiglu":
            if ffn.gate_kind != types.GateKind.SWIGLU or ffn.activation != types.Activation.SILU:
                raise NotImplementedError(
                    "B5: MoE swiglu experts require SwiGLU + SILU (DeepSeek family)"
                )
            if ffn.gate_bias or ffn.up_bias or ffn.down_bias or spec.expert_bias:
                raise NotImplementedError("B5: MoE swiglu experts must have no biases")
        # gpt_oss_clamped_swiglu: any gate_kind/activation is ignored — the
        # clamped SwiGLU math is fixed. We still require fused gate_up
        # layout (per HF's `is_concatenated=False, is_transposed=True`,
        # the underlying tensor is one big [E, H, 2*I] block).
        if ffn.fused_gate_up:
            raise NotImplementedError(
                "B5: MoE expert_ffn.fused_gate_up=True not used by DeepSeek "
                "family — experts.gate_up_proj is packed across experts, "
                "but the per-expert gate/up is two separate halves spliced."
            )
        if spec.group_routing is not None:
            gr = spec.group_routing
            if spec.n_experts % gr.n_groups != 0:
                raise ValueError(
                    f"n_experts ({spec.n_experts}) must divide n_groups "
                    f"({gr.n_groups})"
                )

        self.spec = spec
        self.hidden_size = hidden_size
        self.n_experts = spec.n_experts
        self.top_k = spec.top_k
        self.routed_scaling_factor = spec.routed_scaling_factor
        self.n_shared_experts = spec.n_shared_experts
        self.intermediate_size = ffn.intermediate_size

        # Router gate.
        if spec.router_kind == "topk_then_softmax_with_bias":
            # v6 A3: GPT-OSS router — gate.weight + gate.bias, F.linear,
            # top_k BEFORE softmax (NOT after, unlike V2). Source:
            # modeling_gpt_oss.py:122-135 (GptOssTopKRouter).
            self.gate = _BiasedLinearRouter(
                spec.n_experts, hidden_size, dtype=dtype,
            )
        elif spec.router_kind == "softmax":
            # V2: nn.Linear(hidden, n_experts, bias=False), gate.weight at
            # [n_experts, hidden]. Source: modeling_deepseek_v2.py:90.
            self.gate = nn.Linear(hidden_size, spec.n_experts, bias=False, dtype=dtype)
            self.register_buffer(
                "e_score_correction_bias",
                torch.zeros(0, dtype=torch.float32),
                persistent=False,
            )
        elif spec.router_kind == "hash":
            # v7 P1: DeepSeek-V4 hash router — Parameter `weight` (per-expert
            # scoring head) + Buffer `tid2eid` (token-id → top_k experts).
            # Source: modeling_deepseek_v4.py:1059-1067 (DeepseekV4HashRouter
            # init).
            self.gate = _HashRouter(
                n_experts=spec.n_experts,
                hidden_size=hidden_size,
                vocab_size=spec.hash_vocab_size,
                top_k=spec.top_k,
                dtype=dtype,
            )
        else:  # sigmoid_plus_bias
            # V3 router: a small nn.Module wrapping nn.Parameter weight +
            # buffer e_score_correction_bias. We mirror that exactly so HF
            # state dict loads with the same key names
            # (`gate.weight`, `gate.e_score_correction_bias`).
            # Source: modeling_deepseek_v3.py:139-151.
            self.gate = _SigmoidRouter(spec.n_experts, hidden_size, dtype=dtype)

        # Expert packed tensors.
        # gate_up_proj packs gate (rows 0..I) above up (rows I..2I) per the
        # V2/V3 ordering — verify by chunk(2, dim=-1) in HF:
        # `gate, up = F.linear(state, gate_up_proj[e]).chunk(2, dim=-1)`
        # (modeling_deepseek_v2.py:76 / modeling_deepseek_v3.py:185). Since
        # F.linear(x, W) = x @ W.T, the output is split along the last dim,
        # which corresponds to the FIRST output-axis of W. Therefore in W
        # the gate weights are rows [0, I) and up weights are rows [I, 2I).
        I = ffn.intermediate_size
        # v6 A3: GPT-OSS uses a transposed expert layout:
        #   gate_up_proj  : [E, hidden, 2*I]
        #   gate_up_bias  : [E, 2*I]
        #   down_proj     : [E, I, hidden]
        #   down_bias     : [E, hidden]
        # `is_transposed=True, is_concatenated=False, has_bias=True`.
        # The expert forward does `current_state @ gate_up_proj[e] + bias`
        # rather than `F.linear`. Source: modeling_gpt_oss.py:74-117.
        if spec.expert_kind == "gpt_oss_clamped_swiglu":
            self.experts_gate_up = nn.Parameter(
                torch.empty(spec.n_experts, hidden_size, 2 * I, dtype=dtype)
            )
            self.experts_down = nn.Parameter(
                torch.empty(spec.n_experts, I, hidden_size, dtype=dtype)
            )
            if spec.expert_bias:
                self.experts_gate_up_bias = nn.Parameter(
                    torch.empty(spec.n_experts, 2 * I, dtype=dtype)
                )
                self.experts_down_bias = nn.Parameter(
                    torch.empty(spec.n_experts, hidden_size, dtype=dtype)
                )
            else:
                self.experts_gate_up_bias = None
                self.experts_down_bias = None
        else:
            # Original DeepSeek layout: gate_up_proj [E, 2I, hidden],
            # down_proj [E, hidden, I]. No biases.
            self.experts_gate_up = nn.Parameter(
                torch.empty(spec.n_experts, 2 * I, hidden_size, dtype=dtype)
            )
            self.experts_down = nn.Parameter(
                torch.empty(spec.n_experts, hidden_size, I, dtype=dtype)
            )
            self.experts_gate_up_bias = None
            self.experts_down_bias = None

        # Shared experts (FFN with intermediate = I * n_shared_experts).
        if spec.n_shared_experts > 0:
            shared_ffn = specs.FFNSpec(
                intermediate_size=I * spec.n_shared_experts,
                activation=types.Activation.SILU,
                gate_kind=types.GateKind.SWIGLU,
                fused_gate_up=False,
                gate_bias=False, up_bias=False, down_bias=False,
            )
            self.shared_experts = FeedForward(shared_ffn, hidden_size, dtype=dtype)
        else:
            self.shared_experts = None

    def _route_softmax(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """V2 softmax router. Source: modeling_deepseek_v2.py:100-130.

        Returns (topk_indices, topk_weights) where indices are LongTensor
        [N, top_k] and weights are float32 [N, top_k]. N = B*S.
        """
        # F.linear with fp32-cast inputs/weight — mirrors HF
        # `nn.functional.linear(hidden_states.type(torch.float32),
        #  self.gate.weight.type(torch.float32))`.
        router_logits = F.linear(x.float(), self.gate.weight.float())
        N, E = router_logits.shape
        scores = router_logits.softmax(dim=-1, dtype=torch.float32)

        if self.spec.group_routing is None:
            # Greedy: top-k directly.
            topk_w, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        else:
            gr = self.spec.group_routing
            n_groups = gr.n_groups
            E_per_g = E // n_groups
            # group_scores: per-group MAX (NOT sum) per V2 algorithm.
            group_scores = scores.view(N, n_groups, E_per_g).max(dim=-1).values
            # Top-k_per_group groups by score.
            group_idx = torch.topk(group_scores, k=gr.topk_per_group, dim=-1,
                                   sorted=False).indices
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1.0)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(N, n_groups, E_per_g)
                .reshape(N, E)
            )
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
            topk_w, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1,
                                          sorted=False)

        if self.spec.router_norm:
            denom = topk_w.sum(dim=-1, keepdim=True) + 1e-20
            topk_w = topk_w / denom
        topk_w = topk_w * self.routed_scaling_factor
        return topk_idx, topk_w

    def _route_sigmoid_plus_bias(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V3 sigmoid+bias router. Source: modeling_deepseek_v3.py:214-237.

        Returns (topk_indices, topk_weights). N = B*S.
        """
        # V3 gate.weight: [n_experts, hidden]; identity F.linear in fp32.
        router_logits = self.gate(x)                  # [N, E] fp32
        router_probs = router_logits.sigmoid()
        N, E = router_probs.shape

        if self.spec.group_routing is None:
            # V3 always uses group routing — but supporting None lets us
            # exercise the sigmoid path with a tiny config in tests.
            probs_for_choice = router_probs + self.gate.e_score_correction_bias
            topk_idx = torch.topk(probs_for_choice, k=self.top_k, dim=-1,
                                  sorted=False).indices
        else:
            gr = self.spec.group_routing
            n_groups = gr.n_groups
            E_per_g = E // n_groups
            probs_for_choice = router_probs + self.gate.e_score_correction_bias
            # group_scores: per-group SUM of TOP-2 (V3 algorithm — NOT max).
            group_scores = (
                probs_for_choice.view(N, n_groups, E_per_g)
                .topk(2, dim=-1).values
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=gr.topk_per_group, dim=-1,
                                   sorted=False).indices
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1.0)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(N, n_groups, E_per_g)
                .reshape(N, E)
            )
            # Mask non-selected groups with -inf so top-k skips them.
            scores_for_choice = probs_for_choice.masked_fill(
                ~score_mask.bool(), float("-inf"),
            )
            topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1,
                                  sorted=False).indices

        # CRITICAL: weights are gathered from the BIAS-FREE `router_probs`.
        # Source: modeling_deepseek_v3.py:232.
        topk_w = router_probs.gather(1, topk_idx)
        if self.spec.router_norm:
            denom = topk_w.sum(dim=-1, keepdim=True) + 1e-20
            topk_w = topk_w / denom
        topk_w = topk_w * self.routed_scaling_factor
        return topk_idx, topk_w

    def _route_hash(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """v7 P1: DeepSeek-V4 hash router.

        Source: modeling_deepseek_v4.py:1069-1078.

            logits = F.linear(flat, self.weight)
            scores = self.score_fn(logits)                  # sigmoid
            indices = self.tid2eid[input_ids.reshape(-1)].long()
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            return logits, weights * self.routed_scaling_factor, indices

        `input_ids` must be a LongTensor whose first dim flattens to N (==B*S);
        the routing decision per token is purely deterministic from the
        token id.
        """
        if input_ids is None:
            raise ValueError("router_kind='hash' requires input_ids in forward")
        ids_flat = input_ids.reshape(-1).long()
        N_router = ids_flat.shape[0]
        N_x = x.shape[0]
        if N_router != N_x:
            raise ValueError(
                f"hash routing: input_ids has {N_router} tokens but "
                f"hidden_states has {N_x}; counts must match"
            )
        # Scoring head: sigmoid over per-expert logits. We mirror HF's path
        # exactly — no fp32 upcast (the V4 router runs in the activation
        # dtype unlike V3's sigmoid_plus_bias).
        logits = F.linear(x, self.gate.weight)               # [N, E]
        scores = torch.sigmoid(logits)
        topk_idx = self.gate.tid2eid[ids_flat].long()        # [N, top_k]
        topk_w = scores.gather(1, topk_idx)
        denom = topk_w.sum(dim=-1, keepdim=True) + 1e-20
        topk_w = topk_w / denom
        topk_w = topk_w * self.routed_scaling_factor
        return topk_idx, topk_w

    def _route_topk_then_softmax_with_bias(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """v6 A3: GPT-OSS router.

        Source: modeling_gpt_oss.py:122-135 (GptOssTopKRouter):
            router_logits = F.linear(x, weight, bias)   # gate has BIAS
            top_value, top_idx = topk(router_logits, top_k)
            router_scores = softmax(top_value)          # ONLY over top-k
            return logits, scores, indices
        """
        router_logits = F.linear(
            x, self.gate.weight,
            self.gate.bias if hasattr(self.gate, "bias") else None,
        )                                              # [N, E]
        topk_val, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
        topk_w = torch.softmax(topk_val, dim=-1, dtype=topk_val.dtype)
        # GPT-OSS does not norm beyond softmax and does not scale (no
        # `routed_scaling_factor`).
        return topk_idx, topk_w

    def _experts_forward(
        self,
        x: torch.Tensor,           # [N, hidden]
        topk_idx: torch.Tensor,    # [N, top_k] long
        topk_w: torch.Tensor,      # [N, top_k] float
    ) -> torch.Tensor:
        """Dense-scatter expert dispatch.

        Iterates over each expert that was selected by at least one
        (token, slot) pair. Mirrors the HF reference loop at
        modeling_deepseek_v2.py:64-82 / modeling_deepseek_v3.py:173-191.
        """
        N, H = x.shape
        out = torch.zeros_like(x)
        # expert_mask[e, k, n] = 1 iff token n picked expert e at slot k.
        with torch.no_grad():
            one_hot = F.one_hot(topk_idx, num_classes=self.n_experts)
            expert_mask = one_hot.permute(2, 1, 0)         # [E, top_k, N]
            hit = expert_mask.sum(dim=(-1, -2)).gt(0).nonzero().squeeze(-1)
        for e in hit.tolist():
            slot_pos, tok_idx = torch.where(expert_mask[e])
            current_x = x[tok_idx]                          # [Ne, H]
            gate_up = F.linear(current_x, self.experts_gate_up[e])
            gate, up = gate_up.chunk(2, dim=-1)
            inner = F.silu(gate) * up
            inner = F.linear(inner, self.experts_down[e])   # [Ne, H]
            w = topk_w[tok_idx, slot_pos, None].to(inner.dtype)
            inner = inner * w
            out.index_add_(0, tok_idx, inner.to(out.dtype))
        return out

    def _experts_forward_gpt_oss(
        self,
        x: torch.Tensor,           # [N, hidden]
        topk_idx: torch.Tensor,    # [N, top_k] long
        topk_w: torch.Tensor,      # [N, top_k] float
    ) -> torch.Tensor:
        """v6 A3: GPT-OSS clamped-SwiGLU experts with biased linears.

        Source: modeling_gpt_oss.py:87-117 (GptOssExperts._apply_gate +
        forward loop).

        Per-expert math:
            gate_up = current_state @ gate_up_proj[e] + gate_up_bias[e]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]   # INTERLEAVED
            gate = gate.clamp(max=limit)
            up   = up.clamp(min=-limit, max=limit)
            glu = gate * sigmoid(gate * alpha)
            gated = (up + 1) * glu
            out_e = gated @ down_proj[e] + down_bias[e]
            out += routing_weight * out_e

        Note: in HF the per-expert matmul is `state @ W[e]` (NOT
        F.linear), so W layout is [hidden, 2*I] / [I, hidden] —
        already-transposed.
        """
        N, H = x.shape
        out = torch.zeros_like(x)
        alpha = self.spec.expert_swiglu_alpha
        limit = self.spec.expert_clamp_limit
        with torch.no_grad():
            one_hot = F.one_hot(topk_idx, num_classes=self.n_experts)
            expert_mask = one_hot.permute(2, 1, 0)         # [E, top_k, N]
            hit = expert_mask.sum(dim=(-1, -2)).gt(0).nonzero().squeeze(-1)
        gu_bias = self.experts_gate_up_bias
        d_bias = self.experts_down_bias
        for e in hit.tolist():
            slot_pos, tok_idx = torch.where(expert_mask[e])
            current_x = x[tok_idx]                          # [Ne, H]
            gate_up = current_x @ self.experts_gate_up[e]   # [Ne, 2I]
            if gu_bias is not None:
                gate_up = gate_up + gu_bias[e]
            gate = gate_up[..., ::2]
            up = gate_up[..., 1::2]
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            glu = gate * torch.sigmoid(gate * alpha)
            gated = (up + 1) * glu                          # [Ne, I]
            out_e = gated @ self.experts_down[e]            # [Ne, H]
            if d_bias is not None:
                out_e = out_e + d_bias[e]
            w = topk_w[tok_idx, slot_pos, None].to(out_e.dtype)
            out_e = out_e * w
            out.index_add_(0, tok_idx, out_e.to(out.dtype))
        return out

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [B, S, hidden]
        # input_ids: [B, S] long — REQUIRED for router_kind='hash', ignored
        #   otherwise. v7 P1: source modeling_deepseek_v4.py:1089-1098.
        residuals = x
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])

        if self.spec.router_kind == "softmax":
            topk_idx, topk_w = self._route_softmax(x_flat)
        elif self.spec.router_kind == "sigmoid_plus_bias":
            topk_idx, topk_w = self._route_sigmoid_plus_bias(x_flat)
        elif self.spec.router_kind == "hash":
            topk_idx, topk_w = self._route_hash(x_flat, input_ids)
        else:  # topk_then_softmax_with_bias
            topk_idx, topk_w = self._route_topk_then_softmax_with_bias(x_flat)
        # IMPORTANT: HF runs experts on the BF16/FP32 hidden states (the
        # original input dtype), NOT on the fp32-cast router input. So the
        # gather x[token_idx] is from the original `x_flat` (in x's dtype).
        if self.spec.expert_kind == "gpt_oss_clamped_swiglu":
            routed = self._experts_forward_gpt_oss(x_flat, topk_idx, topk_w)
        else:
            routed = self._experts_forward(x_flat, topk_idx, topk_w)
        routed = routed.reshape(*orig_shape)

        if self.shared_experts is not None:
            routed = routed + self.shared_experts(residuals)
        return routed


class _SigmoidRouter(nn.Module):
    """V3 router gate — Parameter weight + buffer score-correction bias.

    Source: modeling_deepseek_v3.py:139-151.
    """

    def __init__(self, n_experts: int, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_experts, hidden_size, dtype=dtype))
        # HF marks this fp32 via `_keep_in_fp32_modules_strict`; we keep it
        # fp32 to match the additive precision in `probs + bias`.
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(n_experts, dtype=torch.float32),
            persistent=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF: F.linear(x.fp32, weight.fp32) — see modeling_deepseek_v3.py:150.
        return F.linear(x.float(), self.weight.float())


class _HashRouter(nn.Module):
    """v7 P1: DeepSeek-V4 hash router gate.

    Mirrors HF state-dict keys exactly:
        - `gate.weight`: Parameter [n_experts, hidden] — per-expert score head.
        - `gate.tid2eid`: Buffer [vocab_size, top_k] long — frozen token-id →
          expert-id table loaded from checkpoint.

    Source: modeling_deepseek_v4.py:1059-1067.
    """

    def __init__(
        self,
        n_experts: int,
        hidden_size: int,
        vocab_size: int,
        top_k: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_experts, hidden_size, dtype=dtype))
        # tid2eid: persistent buffer carrying the frozen routing table; loaded
        # from the checkpoint at runtime. Initialised to zeros — HF zeros
        # this on init too (see modeling_deepseek_v4.py:1234).
        self.register_buffer(
            "tid2eid",
            torch.zeros(vocab_size, top_k, dtype=torch.long),
            persistent=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Provided for parity with the other gates; the MoE class calls
        # F.linear directly via `_route_hash`. (HF's HashRouter.forward
        # also returns logits, but the production path slices through it.)
        return F.linear(x, self.weight)


class _BiasedLinearRouter(nn.Module):
    """v6 A3: GPT-OSS router — biased nn.Linear-equivalent.

    Source: modeling_gpt_oss.py:122-135 (GptOssTopKRouter).

    Mirrors HF state-dict keys exactly: `gate.weight` (shape [n_experts,
    hidden]) and `gate.bias` (shape [n_experts]).
    """

    def __init__(self, n_experts: int, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_experts, hidden_size, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(n_experts, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
