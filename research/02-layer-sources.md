# 02 - Per-Layer Source Implementations Across Ten SLM Families

## Purpose

This document is the layer-implementation companion to the `llm-layers` API design. The downstream goal is a minimal parameterized API capable of expressing any mainstream SLM transformer block. To find the minimal parameterization, we extract the *same* per-layer computational graph from three independent implementations of each model family:

- **(a) HuggingFace `transformers`** — reference PyTorch
- **(b) `vLLM` `model_executor`** — runtime-optimized PyTorch with tensor parallelism, fused projections, paged KV cache
- **(c) `llama.cpp`** — C++/ggml graph

Three lenses on the same block reveal what is *essential* (present in all three) vs *convention* (e.g. weight packing, attention backend). For each family we list: file paths and line ranges, class names, tensor shapes, pseudocode of `forward`, the quirks (QK-norm, scale variants, biases, gate clamping, etc.), and the divergences between implementations.

All HuggingFace excerpts below are read locally from
`C:\Users\zhengte\external\transformers\src\transformers\models\<name>\modeling_<name>.py`.
vLLM excerpts are from `https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models/<name>.py`.
llama.cpp excerpts are from `https://github.com/ggml-org/llama.cpp/tree/master/src/models/<name>.cpp` plus `src/llama-graph.cpp`.

Tensor-shape conventions used throughout:

```
B  = batch
S  = sequence length (prefill) or 1 (decode step)
D  = hidden_size
H  = num_attention_heads
Hk = num_key_value_heads  (Hk <= H, GQA)
Dh = head_dim             (typically D / H, but Llama-3 8B uses D=4096 H=32 Dh=128)
Df = intermediate_size    (FFN inner)
E  = num_local_experts     (MoE)
Ek = num_experts_per_tok   (top-k)
```

---

## 1. Llama 3

The reference. Every subsequent family is described relative to this baseline.

### 1a. HF transformers — `modeling_llama.py`

File: `transformers/src/transformers/models/llama/modeling_llama.py`
Key lines:

- `LlamaRMSNorm` — lines 52–70
- `LlamaRotaryEmbedding` — lines 73–135
- `LlamaMLP` — lines 171–184
- `LlamaAttention` — lines 224–289
- `LlamaDecoderLayer` — lines 292–332

The decoder layer (HF lines 292–332):

```python
class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm        = LlamaRMSNorm(D, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(D, eps=config.rms_norm_eps)

    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)          # pre-norm
        hidden_states, _ = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states                      # residual 1

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states                      # residual 2
        return hidden_states
```

Attention (HF lines 251–289), pseudocode with shapes:

```python
# hidden_states: [B, S, D]
q = q_proj(h).view(B, S, H,  Dh).transpose(1, 2)     # [B, H,  S, Dh]
k = k_proj(h).view(B, S, Hk, Dh).transpose(1, 2)     # [B, Hk, S, Dh]
v = v_proj(h).view(B, S, Hk, Dh).transpose(1, 2)     # [B, Hk, S, Dh]
q, k = apply_rotary_pos_emb(q, k, cos, sin)          # RoPE, half-rotate variant
k, v = past_key_values.update(k, v, layer_idx)       # KV cache append
# repeat_kv → [B, H, S, Dh] for eager path; SDPA/flash do it internally
attn = softmax((q @ k.T) * head_dim**-0.5 + mask) @ v
out  = attn.transpose(1, 2).reshape(B, S, H*Dh)
return o_proj(out)                                    # [B, S, D]
```

MLP (HF lines 171–184) — SwiGLU:

```python
y = down_proj(silu(gate_proj(x)) * up_proj(x))
# gate_proj, up_proj: D → Df       down_proj: Df → D
```

Quirks of Llama 3 relative to Llama 2 surfaced by the code:

- `config.attention_bias` and `config.mlp_bias` (lines 239–249, 177–179) — Llama-3.x 405B uses biases on some attention projections, so `q_proj/k_proj/v_proj/o_proj` carry an optional bias. The 8B/70B configs leave them False.
- RMSNorm computed in `fp32` then cast back (line 64). Common to most families.
- RoPE `half-rotate` variant (`rotate_half`, line 138). The cos/sin tensor is built `cat(freqs, freqs)` and `q_embed = q*cos + rotate_half(q)*sin`. This is the **non-interleaved** variant; ggml calls this `rope_type=GGML_ROPE_TYPE_NEOX`.
- `scaling = head_dim ** -0.5` baked into the layer at construction.
- GQA via `num_key_value_groups = H // Hk` and `repeat_kv` (line 187). SDPA/flash backends repeat internally.
- The class is decorated with `@use_kernelized_func(apply_rotary_pos_emb)` (line 224) so the rotary op can be swapped for a fused kernel without touching the layer.

### 1b. vLLM — `vllm/model_executor/models/llama.py`

Key changes vs HF:

```python
class LlamaAttention(nn.Module):
    def __init__(self, ...):
        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_size=head_dim,
            total_num_heads=H, total_num_kv_heads=Hk, bias=bias, ...)
        self.o_proj = RowParallelLinear(...)
        self.rotary_emb = get_rope(head_dim, max_position, rope_parameters)
        self.attn = Attention(num_heads, head_dim, scaling,
                              num_kv_heads, cache_config, per_layer_sliding_window=...)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)              # paged KV cache lives inside
        output, _ = self.o_proj(attn_output)
        return output
```

```python
class LlamaMLP(nn.Module):
    def __init__(self, ...):
        self.gate_up_proj = MergedColumnParallelLinear(D, [Df, Df], bias=bias)
        self.down_proj    = RowParallelLinear(Df, D, bias=bias)
        self.act_fn       = SiluAndMul()              # fused SiLU * up
    def forward(self, x):
        x, _ = self.gate_up_proj(x); x = self.act_fn(x); x, _ = self.down_proj(x)
        return x
```

```python
class LlamaDecoderLayer:
    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

Divergences from HF:

1. **Fused QKV** (`QKVParallelLinear`) — one matmul producing `[B*S, (H+2Hk)*Dh]`, then split. Same math, one kernel.
2. **Fused gate-up** (`MergedColumnParallelLinear`) — gate and up live in the *same* weight tensor; `SiluAndMul()` does both halves and the multiply in one kernel.
3. **Residual-passing-through-norm** — the RMSNorm has a `(hidden_states, residual)` two-arg overload that does the *previous* residual add and the norm in one call. This is the "fused add-norm" pattern.
4. **Paged KV cache** — `self.attn = Attention(...)` is a vLLM-specific module that internally uses the paged KV cache; `positions` is the key argument that lets prefill and decode share the same code.
5. **No explicit `past_key_values.update`** call — the cache lives inside `self.attn`.

### 1c. llama.cpp — `src/models/llama.cpp`

Inside the `for il = 0 ... n_layer` loop the per-layer block is:

```cpp
cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
        n_embd_head, n_head, n_head_kv, il);

Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot, rope_type,
        n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
        beta_fast, beta_slow);
Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot, rope_type, ...);

cur = build_attn(inp_attn, model.layers[il].wo, model.layers[il].wo_b,
        Qcur, Kcur, Vcur, ..., il);

ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);    // residual 1

cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
cur = build_ffn(cur, model.layers[il].ffn_up, NULL,
                model.layers[il].ffn_gate, NULL,
                model.layers[il].ffn_down, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);

cur = ggml_add(ctx0, cur, ffn_inp);                    // residual 2
inpL = cur;
```

Notes:

- `build_qkv` reads either three separate tensors (`wq`, `wk`, `wv`) or a fused `wqkv` depending on the GGUF file layout.
- `ggml_rope_ext` is the unified RoPE op; `rope_type` encodes `NEOX` (Llama half-rotate), `NORM` (interleaved GPT-J), `GLM`, or `MROPE`.
- `build_attn` internally does `K_cache = cpy_k(...)`, `V_cache = cpy_v(...)`, and either `ggml_flash_attn_ext` or manual `softmax(QK^T)V`.
- `LLM_FFN_PAR` means parallel gate+up (SwiGLU); `LLM_FFN_SILU` is the activation.

---

## 2. Qwen 3 — QK-Norm

Qwen3 = Llama 3 + per-head RMSNorm on Q and K before RoPE + optional sliding-window alternation.

### 2a. HF — `modeling_qwen3.py`

File: `transformers/src/transformers/models/qwen3/modeling_qwen3.py`
Key lines:

- `Qwen3RMSNorm` 49–67
- `Qwen3Attention` 221–291 (QK-norm at line 248–249, 263–264)
- `Qwen3DecoderLayer` 294–334

Critical excerpt (lines 248–249, 263–264):

```python
# in __init__
self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # only head_dim!
self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

# in forward
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

The QK-norm is *per head_dim*, applied **after** the projection's view-into-heads and **before** RoPE. Per-head_dim norm means `gamma` has shape `[Dh]`, not `[H*Dh]`. This is intentional (the comment on line 249 says "thus post q_norm does not need reshape").

The decoder layer (lines 294–334) is *identical* to Llama's structurally — pre-RMSNorm, attention, residual, post-RMSNorm, MLP, residual. So Qwen3's only block-level deviation is the QK-norm inside attention.

Sliding window is per-layer:
`self.layer_type = config.layer_types[layer_idx]` ∈ `{"full_attention", "sliding_attention"}`. The Qwen3-Next configs alternate; Qwen3-8B currently uses full attention only.

### 2b. vLLM — `vllm/model_executor/models/qwen3.py`

```python
class Qwen3Attention(nn.Module):
    def __init__(self, ...):
        self.qkv_proj = QKVParallelLinear(...)
        self.q_norm = RMSNorm(head_dim, eps=...)      # head_dim only
        self.k_norm = RMSNorm(head_dim, eps=...)
        self.rotary_emb = get_rope(...)
        self.attn = Attention(...)
    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q_by_head = q.view(-1, num_heads_local, head_dim)
        q_by_head = self.q_norm(q_by_head)
        k_by_head = k.view(-1, num_kv_heads_local, head_dim)
        k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(-1, num_heads_local * head_dim)
        k = k_by_head.view(-1, num_kv_heads_local * head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)[0]
```

Same idea: norm is applied per-head, on the head_dim axis. vLLM reshapes explicitly because after `QKVParallelLinear` Q/K live in a flat `[*, H*Dh]` layout.

Inherits `Qwen3MLP` from Qwen2 (which is identical to LlamaMLP).

### 2c. llama.cpp — `src/models/qwen3.cpp`

```cpp
cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
auto [Qcur, Kcur, Vcur] = build_qkv(...);
// QK-norm BEFORE rope
Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);

Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, ...);
Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, ...);
cur  = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur, ..., il);
```

The two extra tensors `attn_q_norm` and `attn_k_norm` are loaded from GGUF; the rest of the block is identical to Llama.

Divergences across HF / vLLM / llama.cpp for Qwen3:
- HF does `view(B,S,H,Dh) → norm on last dim Dh → transpose` (norm acts on `Dh` axis because of the view).
- vLLM does the same but with an explicit reshape because `QKVParallelLinear` outputs a flat 2D tensor `[tokens, H*Dh]`.
- llama.cpp applies norm to the rank-3 GGML tensor with shape `[Dh, H, tokens]`; `build_norm` normalizes the contiguous first axis.

All three define the gamma to be `head_dim`-sized, not `H*Dh`-sized. This is *the* QK-norm convention to standardize in the API.

---

## 3. Gemma 3 — SWA alternation + double norms + softcap

Gemma 3 is the busiest decoder layer of the lot.

### 3a. HF — `modeling_gemma3.py`

File: `transformers/src/transformers/models/gemma3/modeling_gemma3.py`
Key lines:

- `Gemma3RMSNorm` (1+w variant) 128–145
- `Gemma3TextScaledWordEmbedding` 98–109
- `Gemma3MLP` 112–125
- `Gemma3Attention` 306–382 (QK norm 337–338, softcap 333, scaling 317)
- `Gemma3DecoderLayer` 385–428

The decoder layer (HF lines 385–428):

```python
class Gemma3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = Gemma3Attention(config, layer_idx)
        self.mlp       = Gemma3MLP(config)
        self.input_layernorm          = Gemma3RMSNorm(D, eps=eps)
        self.post_attention_layernorm = Gemma3RMSNorm(D, eps=eps)
        self.pre_feedforward_layernorm  = Gemma3RMSNorm(D, eps=eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(D, eps=eps)

    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, ...)
        hidden_states = self.post_attention_layernorm(hidden_states)   # post-attn norm
        hidden_states = residual + hidden_states                        # residual 1

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)  # post-ffn norm
        hidden_states = residual + hidden_states                        # residual 2
        return hidden_states
```

Four RMSNorms per layer (vs Llama's two). The "post" norms run *inside* the residual branch — `residual + norm(sublayer(norm(x)))`. This is the "sandwich norm" pattern from Gemma 2.

Attention quirks:

```python
# line 317
self.scaling = config.query_pre_attn_scalar**-0.5   # NOT head_dim**-0.5
# lines 333–334
self.attn_logit_softcapping = config.attn_logit_softcapping
self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
# lines 337–338
self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=eps)
self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=eps)
# in forward:
query_states = self.q_norm(query_states)
key_states   = self.k_norm(key_states)
```

Softcap is applied inside `eager_attention_forward` (lines 291–294):

```python
if softcap is not None:
    attn_weights = attn_weights / softcap
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * softcap
```

Gemma3 RMSNorm uses the **(1 + w) gain** (lines 137–142):

```python
def forward(self, x):
    output = self._norm(x.float())
    output = output * (1.0 + self.weight.float())  # NOTE: 1 +, not *
    return output.type_as(x)
```

Weights are initialized to zero (so the initial gain is 1.0). HF init code at line 462: `init.zeros_(module.weight)`.

Input embedding is scaled by `sqrt(D)` (line 500):

```python
self.embed_tokens = Gemma3TextScaledWordEmbedding(
    vocab_size, D, padding_idx, embed_scale=self.config.hidden_size**0.5)
```

SWA alternation is configured per-layer via `config.layer_types[layer_idx] in {"full_attention", "sliding_attention"}`. Gemma 3 publishes a 5-sliding-then-1-global pattern (5:1) so the model layer index parity determines which mask gets used.

### 3b. vLLM — `vllm/model_executor/models/gemma3.py`

vLLM mirrors the four-norm DecoderLayer and instantiates `q_norm`, `k_norm`. Attention uses `query_pre_attn_scalar**-0.5` as scaling, `attn_logits_soft_cap` propagated to the attention backend (FlashAttention has a `logits_soft_cap` argument; SDPA falls back to manual). SWA per layer is set via `per_layer_sliding_window` kwarg of `Attention(...)`.

### 3c. llama.cpp — `src/models/gemma3.cpp`

```cpp
cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
auto [Qcur, Kcur, Vcur] = build_qkv(...);
Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
Qcur = ggml_rope_ext(ctx0, Qcur, ..., freq_base_swa or freq_base, ...);
Kcur = ggml_rope_ext(ctx0, Kcur, ..., freq_base_swa or freq_base, ...);
cur  = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur,
                  hparams.f_attention_scale,      // = query_pre_attn_scalar**-0.5
                  il);
cur  = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);    // residual 1

cur = build_norm(sa_out, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
cur = build_ffn(cur, model.layers[il].ffn_up, ..., model.layers[il].ffn_gate, ...,
                model.layers[il].ffn_down, ..., LLM_FFN_GELU, LLM_FFN_PAR, il);
cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, -1);
cur = ggml_add(ctx0, cur, sa_out);                   // residual 2
```

Six norm tensors per layer in GGUF: `attn_norm`, `attn_q_norm`, `attn_k_norm`, `attn_post_norm`, `ffn_norm`, `ffn_post_norm`. The `1 + w` reparametrization happens inside ggml's RMSNorm fused kernel by storing `w` and adding 1 inside; some converters bake `1+w` into the saved weights and pass plain RMS. (See [llama.cpp #29402 discussion](https://github.com/huggingface/transformers/pull/29402)). The HF code path always computes `(1 + w)` at runtime.

SWA alternation: llama.cpp tracks `swa_type=STANDARD` for Gemma and computes the mask once per template; per-layer it just sets `freq_base_swa` and the layer's mask binding.

Activation: Gemma uses **GELU** (`hidden_activation = "gelu_pytorch_tanh"`), not SiLU like the rest. The MLP is structurally identical to SwiGLU but with GELU as the gating function:
`y = down_proj(gelu(gate_proj(x)) * up_proj(x))`.

---

## 4. Phi-3 — Fused QKV, Fused gate-up, Partial RoPE

### 4a. HF — `modeling_phi3.py`

File: `transformers/src/transformers/models/phi3/modeling_phi3.py`
Key lines:

- `Phi3MLP` 49–64 (fused gate-up)
- `Phi3Attention` 208–271 (fused qkv_proj, partial RoPE)
- `Phi3DecoderLayer` 295–335 (residual dropouts)
- `apply_rotary_pos_emb` 178–205 (partial rotary)

Fused gate-up MLP (lines 49–64):

```python
class Phi3MLP(nn.Module):
    def __init__(self, config):
        self.gate_up_proj = nn.Linear(D, 2*Df, bias=False)
        self.down_proj    = nn.Linear(Df, D,  bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]   # silu

    def forward(self, hidden_states):
        up_states = self.gate_up_proj(hidden_states)          # [B, S, 2*Df]
        gate, up_states = up_states.chunk(2, dim=-1)          # [B, S, Df] each
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)
```

Note the *order* of the chunks: `gate, up = chunk(2, -1)` — gate is the **first** half, up is the **second** half. Some MoE implementations have the opposite convention. HF/Phi-3 GGUF conversion code has to know which half is which.

Fused QKV (lines 222–224):

```python
op_size = H*Dh + 2*Hk*Dh   # query + key + value packed
self.qkv_proj = nn.Linear(D, op_size, bias=False)
self.o_proj   = nn.Linear(H*Dh, D, bias=False)
```

In forward (lines 237–245):

```python
qkv = self.qkv_proj(hidden_states)
query_pos = H*Dh
query_states = qkv[..., :query_pos]
key_states   = qkv[..., query_pos : query_pos + Hk*Dh]
value_states = qkv[..., query_pos + Hk*Dh :]
```

Partial rotary (lines 196–205):

```python
cos = cos.unsqueeze(unsqueeze_dim); sin = sin.unsqueeze(unsqueeze_dim)
rotary_dim = cos.shape[-1]                       # = head_dim * partial_rotary_factor
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
```

Some Phi-3 variants set `partial_rotary_factor = 0.5` — half of each head's dims get RoPE, the rest pass through unchanged. The inv_freq for the cos/sin table is computed on `int(head_dim * partial_rotary_factor)` (lines 106–108).

Decoder layer (lines 295–335) — main diffs vs Llama are the **residual dropouts**:

```python
self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
self.resid_mlp_dropout  = nn.Dropout(config.resid_pdrop)
...
hidden_states = residual + self.resid_attn_dropout(hidden_states)   # diff w/ Llama
hidden_states = residual + self.resid_mlp_dropout(hidden_states)    # diff w/ Llama
```

The dropouts default to `0.0` in inference so they're no-ops, but training graphs need them. Also note: a Phi-3-mini-128k variant uses LongRoPE — same `apply_rotary_pos_emb` with a `rope_scaling = {"type":"longrope", "short_factor":..., "long_factor":...}`, handled in `Phi3RotaryEmbedding.__init__` lines 75–84.

### 4b. vLLM — `vllm/model_executor/models/phi3.py`

vLLM's phi3.py is a *one-liner*: `Phi3ForCausalLM` inherits from `LlamaForCausalLM` with `packed_modules_mapping = {"qkv_proj": ["qkv_proj"], "gate_up_proj": ["gate_up_proj"]}`. The fused projections are literally the same modules as Llama's `QKVParallelLinear` and `MergedColumnParallelLinear`. The packing map tells vLLM's weight loader to map a single GGUF/HF tensor name to one fused weight rather than three/two separate ones.

### 4c. llama.cpp — `src/models/phi3.cpp`

`create_tensor_qkv(...)` loads either separate `wq/wk/wv` (for HF safetensors) or a fused `wqkv` (for converted GGUF) — the code handles both. Partial RoPE is encoded via `n_rot` being smaller than `n_embd_head`. FFN uses `LLM_FFN_SILU` with `LLM_FFN_PAR` for SwiGLU; the GGUF stores `ffn_up` of shape `[D, 2*Df]` when fused.

---

## 5. Mistral — Sliding window everywhere

Mistral is essentially "Llama with SWA on every layer". Mixtral adds MoE.

### 5a. HF — `modeling_mistral.py`

File: `transformers/src/transformers/models/mistral/modeling_mistral.py`
Key lines:

- `MistralAttention` 122–178
- `MistralDecoderLayer` 202–240

The decoder layer is *structurally identical to Llama's* (pre-norm, attention, residual, post-norm, MLP, residual). The only block-level diff is in `MistralAttention.forward` (line 172):

```python
attn_output, attn_weights = attention_interface(
    self, query_states, key_states, value_states, attention_mask,
    dropout=..., scaling=self.scaling,
    sliding_window=getattr(self.config, "sliding_window", None),   # diff w/ Llama
    **kwargs)
```

`config.sliding_window` (e.g. 4096) is propagated to every layer, and the attention backend (SDPA/FlashAttention) uses it to mask out tokens older than the window. There's no per-layer alternation in Mistral; SWA is global.

The model-level `forward` (line 372) chooses the mask:

```python
mask_function = create_causal_mask if self.config.sliding_window is None \
                else create_sliding_window_causal_mask
```

### 5b. vLLM — `vllm/model_executor/models/mistral.py`

vLLM's Mistral derives from its Llama implementation, with sliding window propagated through `per_layer_sliding_window`. Recent vLLM also has a `MistralAttention` that adds the Llama-4 (and Mistral-Large) **position-dependent scaling**:

```
scale = 1 + beta * log(1 + floor(positions / original_max_position_embeddings))
q = q * scale
```

For canonical Mistral-7B / Mixtral 8x7B this falls back to no scaling.

### 5c. llama.cpp — `src/models/mistral4.cpp` (and earlier mistral graphs)

Currently llama.cpp shares the Llama graph builder for Mistral 7B because the only deviation is SWA, which is configured by the KV cache type. The cache is per-layer and capped at `n_swa` tokens for sliding layers. RoPE uses `freq_base = 1e6` (Mistral) or `1e7` (some checkpoints) vs Llama's `5e5/1e6`.

---

## 6. DeepSeek-V3 — Multi-head Latent Attention (MLA) + Auxiliary-free MoE

The single most distinctive layer in this list.

### 6a. HF — `modeling_deepseek_v3.py`

File: `transformers/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py`
Key lines:

- `DeepseekV3MLP` 123–136 (standard SwiGLU, used as shared experts)
- `DeepseekV3TopkRouter` 139–151
- `DeepseekV3NaiveMoe` 154–191
- `DeepseekV3MoE` 194–247 (group-limited routing)
- `DeepseekV3Attention` 364–479 (MLA)
- `DeepseekV3DecoderLayer` 482–526

#### MLA Attention

Parameters:

```
q_lora_rank        = 1536           # Q LoRA rank (None for "lite" variants)
kv_lora_rank       =  512           # KV LoRA rank
qk_nope_head_dim   =  128           # non-rope portion of QK head
qk_rope_head_dim   =   64           # rope portion of QK head
v_head_dim         =  128           # value head dim
qk_head_dim        = qk_nope_head_dim + qk_rope_head_dim = 192
num_heads          =  128
```

`__init__` (lines 367–414):

```python
# Q path: low-rank
if self.q_lora_rank is None:
    self.q_proj = nn.Linear(D, H*qk_head_dim, bias=False)
else:
    self.q_a_proj      = nn.Linear(D, q_lora_rank, bias=attention_bias)
    self.q_a_layernorm = DeepseekV3RMSNorm(q_lora_rank)
    self.q_b_proj      = nn.Linear(q_lora_rank, H*qk_head_dim, bias=False)

# KV path: low-rank + decoupled rope key
self.kv_a_proj_with_mqa = nn.Linear(D, kv_lora_rank + qk_rope_head_dim, bias=attention_bias)
self.kv_a_layernorm     = DeepseekV3RMSNorm(kv_lora_rank)
self.kv_b_proj          = nn.Linear(kv_lora_rank, H*(qk_nope_head_dim + v_head_dim), bias=False)

self.o_proj = nn.Linear(H*v_head_dim, D, bias=attention_bias)
self.scaling = qk_head_dim ** (-0.5)
# YaRN mscale (lines 408-414):
if rope_type != "default":
    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    self.scaling = self.scaling * mscale * mscale
```

`forward` (lines 416–479):

```python
q_states = q_b_proj(q_a_layernorm(q_a_proj(h)))         # [B, S, H * qk_head_dim]
q_states = q_states.view(B, S, H, qk_head_dim).transpose(1,2)  # [B, H, S, qk_head_dim]
q_pass, q_rot = torch.split(q_states, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

compressed_kv = kv_a_proj_with_mqa(h)                   # [B, S, kv_lora_rank + qk_rope_head_dim]
k_pass, k_rot = torch.split(compressed_kv, [kv_lora_rank, qk_rope_head_dim], dim=-1)

k_pass = kv_b_proj(kv_a_layernorm(k_pass))              # [B, S, H * (qk_nope_head_dim + v_head_dim)]
k_pass = k_pass.view(B, S, H, qk_nope_head_dim + v_head_dim).transpose(1,2)
k_pass, value_states = torch.split(k_pass, [qk_nope_head_dim, v_head_dim], dim=-1)

k_rot  = k_rot.view(B, 1, S, qk_rope_head_dim)          # shared key-rope across heads!
cos, sin = position_embeddings
q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
k_rot  = k_rot.expand(B, H, S, qk_rope_head_dim)

query_states = torch.cat((q_pass, q_rot), dim=-1)       # [B, H, S, qk_head_dim]
key_states   = torch.cat((k_pass, k_rot), dim=-1)       # [B, H, S, qk_head_dim]

# Cache update — note the cached values store qk_head_dim keys, v_head_dim values
key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)

# Flash attention requires equal-dim Q/K/V, so pad V if needed (lines 456-457)
if flash_requested and qk_head_dim != v_head_dim:
    value_states = F.pad(value_states, [0, qk_head_dim - v_head_dim])

attn_output, _ = attention_interface(self, query_states, key_states, value_states,
                                      attention_mask, scaling=self.scaling, ...)

if flash_requested and qk_head_dim != v_head_dim:
    attn_output = attn_output[..., :v_head_dim]   # unpad

attn_output = attn_output.reshape(B, S, H * v_head_dim).contiguous()
return self.o_proj(attn_output), _
```

Key insights:

- The cached representation is **already decompressed** here (`key_states` and `value_states` after cat). That's HF's simple-correct path. Production MLA inference (vLLM, llama.cpp) caches the **compressed** `compressed_kv` (`[kv_lora_rank + qk_rope_head_dim]`) and absorbs `kv_b_proj` into `o_proj` to reduce KV-cache size by ~10x.
- `qk_rope_head_dim` is shared across all heads (`view(B, 1, S, qk_rope_head_dim)` then `expand`).
- `q_pass / q_rot` split is **inside each head**, not across heads.

#### Decoder layer (482–526)

```python
if layer_idx >= config.first_k_dense_replace:
    self.mlp = DeepseekV3MoE(config)
else:
    self.mlp = DeepseekV3MLP(config)
```

The first `first_k_dense_replace` layers (typically 1 or 3) are *dense* SwiGLU; the rest are MoE. The pre-norm/post-norm pattern is identical to Llama otherwise.

#### MoE block (194–247)

```python
class DeepseekV3MoE(nn.Module):
    def __init__(self, config):
        self.experts = DeepseekV3NaiveMoe(config)    # routed experts
        self.gate    = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(config,
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts)
        # group routing params
        self.n_group, self.topk_group = config.n_group, config.topk_group
        self.norm_topk_prob   = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()                # sigmoid, not softmax!
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        # group-limited routing:
        group_scores = router_logits_for_choice.view(-1, n_group, n_routed_experts // n_group) \
                            .topk(2, dim=-1)[0].sum(dim=-1)
        group_idx  = torch.topk(group_scores, k=topk_group, dim=-1)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1) \
                       .expand(-1, n_group, n_routed_experts // n_group) \
                       .reshape(-1, n_routed_experts)
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), -inf)
        topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, D)
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)   # shared expert add
        return hidden_states
```

Unique to DeepSeek-V3:

- **Sigmoid gating** instead of softmax (line 215). The "aux-loss-free" load balancer relies on `e_score_correction_bias` (line 216) — a *non-trainable* bias updated outside backprop to balance experts.
- **Group-limited routing** (lines 217–229): experts are arranged into `n_group` groups; pick `topk_group` groups, then `top_k` experts among those groups. Reduces inter-node communication.
- **Shared experts**: a tiny always-on FFN (`shared_experts`) added to every token's MoE output. Acts as a guaranteed-good baseline.
- **`routed_scaling_factor`** (typically 2.5) — boosts routed expert contributions.

### 6b. vLLM — `vllm/model_executor/models/deepseek_v2.py`

vLLM has `DeepseekV2MLAAttention` that supports caching the *compressed* form: `fused_qkv_a_proj` outputs `[q_lora_rank, kv_lora_rank + qk_rope_head_dim]` in one matmul, the `kv_a_layernorm` is applied to the kv-compressed part, and KV cache stores `compressed_kv` (size `kv_lora_rank + qk_rope_head_dim`). The dot-product attention then uses an **absorbed** form: `kv_b_proj` weights are merged into `o_proj` during weight loading. vLLM's MLA decode path uses `flash_mla` or `triton_mla` kernels that consume compressed K and reconstruct on-the-fly.

The MoE is implemented via `FusedMoE` with `topk_method="noaux_tc"` (no-aux top-c, the sigmoid + group routing variant), `n_shared_experts` plumbed in as a separate MLP added to the output.

### 6c. llama.cpp — `src/models/deepseek2.cpp`

```cpp
// Q path
if (n_lora_q > 0) {
    q_proj = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
    q_proj = build_norm(q_proj, model.layers[il].attn_q_a_norm, NULL, LLM_NORM_RMS, il);
    q_proj = ggml_mul_mat(ctx0, model.layers[il].wq_b, q_proj);
} else {
    q_proj = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
}
// q reshape to {qk_nope_head_dim+qk_rope_head_dim, n_head, n_tokens}, split into q_nope, q_rope
// k path (compressed)
ggml_tensor * kv_compressed_with_k_rope = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
// split kv_compressed (kv_lora_rank) and k_rope (qk_rope_head_dim)
kv_compressed = build_norm(kv_compressed, model.layers[il].attn_kv_a_norm, NULL, LLM_NORM_RMS, il);
ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
// reshape kv to {qk_nope_head_dim+v_head_dim, n_head, n_tokens}, split into k_nope, v
// rope on q_rope and k_rope only
q_rope = ggml_rope_ext(ctx0, q_rope, inp_pos, ...);
k_rope = ggml_rope_ext(ctx0, k_rope, inp_pos, ...);
// concatenate q = (q_nope, q_rope), k = (k_nope, k_rope_broadcast)
cur = build_attn(inp_attn, model.layers[il].wo, NULL, q, k, v, kq_scale_with_mscale, il);

// FFN
if (il < n_layer_dense_lead) {
    // dense SwiGLU
    cur = build_ffn(cur, ffn_up, NULL, ffn_gate, NULL, ffn_down, NULL, NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
} else {
    // MoE: routed + shared
    moe_out = build_moe_ffn(cur, ffn_gate_inp, ffn_up_exps, ffn_gate_exps, ffn_down_exps,
                            n_expert, n_expert_used, LLM_FFN_SILU, /*norm_topk*/ true,
                            expert_weights_scale, gating_func_type, il);
    ffn_shexp = build_ffn(cur, ffn_up_shexp, NULL, ffn_gate_shexp, NULL,
                          ffn_down_shexp, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cur = ggml_add(ctx0, moe_out, ffn_shexp);
}
```

Divergence: HF caches *decompressed* K/V (size `H*(qk_head_dim + v_head_dim)`); vLLM and llama.cpp cache **compressed** `kv_lora_rank + qk_rope_head_dim` and absorb `kv_b_proj` into the attention output path. For the API, this means `kv_b_proj` must be an *optional* layer that may be absorbed.

---

## 7. Mixtral / Qwen3-MoE — Plain dense + sparse pattern

### 7a. HF Mixtral — `modeling_mixtral.py`

File: `transformers/src/transformers/models/mixtral/modeling_mixtral.py`
Key lines:

- `MixtralExperts` 61–98 (3D experts)
- `MixtralTopKRouter` 101–116
- `MixtralSparseMoeBlock` 119–135
- `MixtralAttention` 294–351 (identical to Mistral, including SWA)
- `MixtralDecoderLayer` 354–389

Router (lines 101–116):

```python
router_logits = F.linear(hidden_states, self.weight)         # (T, E)
router_probs  = F.softmax(router_logits.float(), dim=-1)
router_top_value, router_indices = torch.topk(router_probs, top_k, dim=-1)
router_top_value /= router_top_value.sum(dim=-1, keepdim=True)   # renormalize
```

Softmax-then-topk-then-renorm. Contrast DeepSeek-V3: sigmoid-then-topk-with-bias-then-(optionally)-renorm.

Experts (lines 61–98) — fused 3D weights:

```python
self.gate_up_proj = nn.Parameter(torch.empty(E, 2*Df, D))
self.down_proj    = nn.Parameter(torch.empty(E, D, Df))
```

The naive loop iterates `expert_hit` then `index_add_`s the result. vLLM and inference engines replace this with `FusedMoE` kernels (Triton / CUTLASS), but the **weight layout** stays the same: a 3D tensor `[E, 2*Df, D]` for fused gate-up.

DecoderLayer (lines 354–389) is structurally Llama; `self.mlp = MixtralSparseMoeBlock(config)` is the only swap.

Mixtral attention propagates `sliding_window=getattr(self.config, "sliding_window", None)` — the original Mixtral 8x7B had `sliding_window=4096` but the standard recipe ignores it.

### 7b. vLLM — `vllm/model_executor/models/mixtral.py`

Uses `FusedMoE` with `renormalize=True`, `top_k=2` (default). Gate is `ReplicatedLinear(D, E)` (small, broadcast not sharded). Expert parallel (EP) groups distribute experts across ranks.

### 7c. llama.cpp — `src/models/mixtral.cpp`-style (shares a generic LLM_ARCH_LLAMA-MoE path)

```cpp
cur = build_moe_ffn(cur, ffn_gate_inp, ffn_up_exps, ffn_gate_exps, ffn_down_exps,
                    n_expert, n_expert_used, LLM_FFN_SILU, /*norm_topk*/ true,
                    /*expert_weights_scale*/ 0.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
```

The 3D expert weights live as `ffn_*_exps` GGUF tensors with shape `[D, Df, E]` (transposed from PyTorch layout).

**Qwen3-MoE** is the same pattern: `Qwen3MoeSparseMoeBlock` = router (softmax+topk+renorm) + 3D experts; `Qwen3MoeDecoderLayer` = Qwen3 attention (with QK-norm) + MoE FFN. Some layers are dense Qwen3MLP, configured by `mlp_only_layers` (a list of layer indices that stay dense). Qwen3-Next further adds the linear-attention alternation, which we don't enumerate here.

---

## 8. Granite — Residual / Embedding / Attention / Logits multipliers

Granite (1B–3B and the larger 3.1) is "Llama with four scalar multipliers".

### 8a. HF — `modeling_granite.py`

File: `transformers/src/transformers/models/granite/modeling_granite.py`
Key lines:

- `GraniteAttention` 115–179 (`self.scaling = config.attention_multiplier`, line 124)
- `GraniteDecoderLayer` 219–280 (`residual_multiplier`, lines 228, 273, 278)
- `GraniteModel` 368–443 (`embedding_multiplier`, line 405)
- `GraniteForCausalLM` 446–504 (`logits_scaling`, line 504)

The decoder layer (HF lines 230–280):

```python
def forward(self, hidden_states, ...):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(hidden_states, ...)
    hidden_states = residual + hidden_states * self.residual_multiplier   # diff w/ Llama

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states * self.residual_multiplier   # diff w/ Llama
    return hidden_states
```

Attention scaling (line 124):

```python
self.scaling = config.attention_multiplier      # NOT head_dim**-0.5
```

Model forward (line 405):

```python
inputs_embeds = inputs_embeds * self.embedding_multiplier
```

LM head (line 504):

```python
logits = logits / self.config.logits_scaling    # main diff with Llama
```

For Granite 3.1-3B-Instruct values are roughly:

```
attention_multiplier  = 0.0078125  ≈ 1/(8*16)  (instead of 1/sqrt(64) = 0.125)
residual_multiplier   = 0.22
embedding_multiplier  = 12.0
logits_scaling        = 8.0
```

These are **muP-style scale factors**, allowing hyper-parameter transfer from a small "anchor" model.

### 8b. vLLM — `vllm/model_executor/models/granite.py`

vLLM mirrors HF: scaling applied at the same four locations, with `residual_multiplier` baked into the `RMSNorm`-residual fused op when supported. `attention_multiplier` propagated to `Attention(...)` as the `scaling` kwarg.

### 8c. llama.cpp — `src/models/granite.cpp`

```cpp
const float kq_scale = hparams.f_attention_scale == 0.0f
    ? 1.0f / sqrtf(float(n_embd_head))
    : hparams.f_attention_scale;
// after attention sublayer:
if (hparams.f_residual_scale) {
    cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
}
ffn_inp = ggml_add(ctx0, cur, inpSA);
// after FFN:
if (hparams.f_residual_scale) {
    cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
}
cur = ggml_add(ctx0, cur, ffn_inp);
// final logits:
cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
```

`f_attention_scale`, `f_residual_scale`, `f_embedding_scale`, `f_logit_scale` are GGUF hyperparams. llama.cpp does the embedding scale inside `build_inp_embd`. All four are scalar multipliers — exactly the parameterization the API should expose.

---

## 9. OLMo 2 — Post-Norm

### 9a. HF — `modeling_olmo2.py`

File: `transformers/src/transformers/models/olmo2/modeling_olmo2.py`
Key lines:

- `Olmo2Attention` 206–276 (full-`H*Dh` Q/K-norm, lines 231–232)
- `Olmo2DecoderLayer` 295–333

Decoder layer (HF lines 295–333):

```python
class Olmo2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = Olmo2Attention(config, layer_idx)
        self.mlp       = Olmo2MLP(config)
        # NOTE: no input_layernorm
        self.post_attention_layernorm  = Olmo2RMSNorm(D, eps=eps)
        self.post_feedforward_layernorm = Olmo2RMSNorm(D, eps=eps)

    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states, _ = self.self_attn(hidden_states, ...)
        hidden_states = self.post_attention_layernorm(hidden_states)   # post-norm
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states) # post-norm
        hidden_states = residual + hidden_states
        return hidden_states
```

Compare Gemma 3 (sandwich norm = pre-norm + post-norm + residual outside) vs Llama (pre-norm + residual outside) vs OLMo 2 (no pre-norm, post-norm wraps the sublayer output, residual outside). The difference is:

- Llama: `x_out = x + sublayer(norm(x))`
- Gemma3: `x_out = x + post_norm(sublayer(pre_norm(x)))`
- OLMo2: `x_out = x + post_norm(sublayer(x))`

The OLMo2 Q/K-norm (lines 231–232) is on the **full** `H*Dh` dim, not per-head:

```python
self.q_norm = Olmo2RMSNorm(config.num_attention_heads * self.head_dim, ...)
self.k_norm = Olmo2RMSNorm(config.num_key_value_heads * self.head_dim, ...)
```

— gamma is `[H*Dh]`. Contrast Qwen3/Gemma3, which use `[Dh]` per-head. Both are valid; OLMo2 chose the broader form. Source comment in Qwen3 (line 248): "unlike olmo, only on the head dim!" — confirming this is a real divergence.

### 9b. vLLM — `vllm/model_executor/models/olmo2.py`

Same forward structure. `q_norm` and `k_norm` sized `H*Dh / Hk*Dh` (matching HF). Post-norm placement preserved.

### 9c. llama.cpp — `src/models/olmo2.cpp`

```cpp
// no pre-attention norm; layer input goes directly into QKV
auto [Qcur, Kcur, Vcur] = build_qkv(...);
Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
Qcur = ggml_rope_ext(ctx0, Qcur, ...);  Kcur = ggml_rope_ext(...);
cur = build_attn(...);
cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
cur = ggml_add(ctx0, cur, inpSA);

cur = build_ffn(cur, ffn_up, NULL, ffn_gate, NULL, ffn_down, NULL, NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);
cur = ggml_add(ctx0, cur, ffn_inp);
```

Confirms: no pre-norm tensors are loaded; `attn_q_norm`, `attn_k_norm`, `attn_post_norm`, `ffn_post_norm` are the four norm gammas.

---

## 10. Jamba — Hybrid attention + Mamba SSM (+ optional MoE)

The non-pure-transformer reference.

### 10a. HF — `modeling_jamba.py`

File: `transformers/src/transformers/models/jamba/modeling_jamba.py`
Key lines:

- `JambaAttention` 148–200
- `JambaMambaMixer` 202–474 (huge)
- `JambaMLP` 477–492
- `JambaSparseMoeBlock` 533–567
- `JambaAttentionDecoderLayer` 570–605
- `JambaMambaDecoderLayer` 608–638

#### Two decoder layer classes

```python
class JambaAttentionDecoderLayer:
    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)   # MLP or MoE
        return residual + hidden_states

class JambaMambaDecoderLayer:
    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states=hidden_states,
                                    cache_params=past_key_values,
                                    attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        return residual + hidden_states
```

Alternation is config-driven: `config.layers_block_type[layer_idx] in {"attention", "mamba"}`. The full Jamba 52B alternates `1 attention : 7 mamba`, with MoE applied to a subset of FFNs (configured by `config.layers_num_experts[layer_idx] > 1`).

`feed_forward` is either `JambaMLP` (standard SwiGLU) or `JambaSparseMoeBlock` (top-2 router + experts), chosen per layer.

#### MambaMixer (a sketch — lines 202–474)

```python
class JambaMambaMixer(nn.Module):
    def __init__(self, config, layer_idx):
        self.intermediate_size = config.mamba_expand * D     # E_int = 2*D typically
        self.ssm_state_size    = config.mamba_d_state        # N (usually 16)
        self.time_step_rank    = config.mamba_dt_rank
        self.conv1d = nn.Conv1d(E_int, E_int, kernel_size=4, groups=E_int, padding=3)
        self.in_proj  = nn.Linear(D, E_int*2, bias=False)     # → (x, z)
        self.x_proj   = nn.Linear(E_int, dt_rank + 2*N, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, E_int, bias=True)
        self.out_proj = nn.Linear(E_int, D, bias=False)
        self.A_log    = nn.Parameter(torch.empty(E_int, N))   # log-A
        self.D        = nn.Parameter(torch.ones(E_int))

    def forward(self, hidden_states, cache_params, attention_mask):
        # 1) project to (x, gate=z)
        proj = self.in_proj(hidden_states).transpose(1,2)     # [B, 2*E_int, S]
        x, z = proj.chunk(2, dim=1)
        # 2) causal conv (depthwise)
        x = causal_conv1d_fn(x, self.conv1d.weight.squeeze(1), self.conv1d.bias, "silu")
        # 3) selective scan parameters
        ssm_params = self.x_proj(x.transpose(1,2))            # [B, S, dt_rank + 2N]
        dt, B_, C_ = torch.split(ssm_params, [dt_rank, N, N], dim=-1)
        dt = self.dt_proj(dt).transpose(1,2)                   # [B, E_int, S]
        A = -torch.exp(self.A_log.float())                     # [E_int, N]
        # 4) selective scan
        y = selective_scan_fn(x, dt, A, B_.transpose(1,2), C_.transpose(1,2),
                               D=self.D, z=z, delta_softplus=True)
        # 5) output projection
        return self.out_proj(y.transpose(1,2))
```

Key tensors (the SSM-equivalent of QKV):

- `in_proj` (D → 2·E_int) — analogous to fused gate-up in MLP
- `conv1d` (depthwise, kernel 4)
- `x_proj` (E_int → dt_rank + 2·N) — projects x to ssm params
- `dt_proj` (dt_rank → E_int) — projects dt to per-channel time step
- `A_log`, `D` — SSM state matrix and skip-connection
- `out_proj` (E_int → D)

The state lives in the "Mamba KV cache": `conv_state` (B, E_int, 4) and `ssm_state` (B, E_int, N).

### 10b. vLLM — `vllm/model_executor/models/jamba.py`

Uses `MambaMixer` from `vllm/model_executor/layers/mamba/mamba_mixer.py` (CUDA fast path). Same layer alternation logic: `JambaAttentionDecoderLayer` and `JambaMambaDecoderLayer`. KV cache management is hybrid — attention layers use paged KV cache; mamba layers use a separate per-layer state tensor.

### 10c. llama.cpp — `src/models/jamba.cpp`

```cpp
// per-layer dispatch
if (n_head_kv(il) == 0) {              // Mamba layer
    cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
    cur = build_mamba_layer(inp_recr, cur, model, ubatch, il);
} else {                                // attention layer
    cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
    auto [Q, K, V] = build_qkv(...);
    Q = ggml_rope_ext(...); K = ggml_rope_ext(...);
    cur = build_attn(inp_attn, model.layers[il].wo, NULL, Q, K, V, ..., il);
}
cur = ggml_add(ctx0, cur, inpL);   // residual after attention/mamba

// FFN (MLP or MoE)
cur = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
if (model.layers[il].ffn_gate_inp) {
    cur = build_moe_ffn(cur, ...);
} else {
    cur = build_ffn(cur, ffn_up, NULL, ffn_gate, NULL, ffn_down, NULL, NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
}
cur = ggml_add(ctx0, cur, ffn_inp);
```

The trick: `n_head_kv(il) == 0` is the GGUF convention to mark a Mamba layer. `build_mamba_layer` does its own state-recurrent attention; the SSM state lives in `inp_recr` (a recurrent-cache binding).

---

## Universal subset — the API floor

After scanning ten families, the operations that appear in **every single** decoder layer are:

1. **Residual stream** of shape `[B, S, D]`.
2. **Token-wise normalization** of the stream into the residual (only the *placement* varies: pre, post, or sandwich).
3. **Token mixer** (attention OR an SSM) producing `[B, S, D]`, added with residual.
4. **Channel mixer** (MLP/MoE) producing `[B, S, D]`, added with residual.
5. For attention:
   - 4 linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), possibly fused as `qkv_proj`.
   - A position-encoding hook (`apply_rotary_pos_emb` or a no-op for SSM layers).
   - A KV-cache hook (`past_key_values.update` or its paged equivalent).
   - A scaled-dot-product call (`softmax(QK^T / s) @ V` or a flash equivalent).
6. For the channel mixer:
   - Gate-Up-Down (SwiGLU = `down(act(gate(x)) * up(x))`) or fused `gate_up_proj`.
   - Optional MoE wrapper: router (linear → top-k) + expert FFN bank + (optional) shared experts.
7. **Norm hyperparameter**: `rms_norm_eps` (or `layer_norm_eps`).
8. **Cache key**: `layer_idx` for indexing per-layer KV state.

Every implementation passes the same five tensors through the layer: `hidden_states`, `attention_mask`, `position_embeddings` (or `positions`), `past_key_values` (or `kv_cache`), and a per-layer index.

---

## Variant axes — the API parameter space

The minimal API must let a config switch on these axes:

1. **Norm placement** — `{pre, post, sandwich}` × `{attention, ffn}` (4 booleans collapse to one enum per sublayer). Llama=pre, OLMo2=post, Gemma3=sandwich, Granite=pre.

2. **Norm formula** — `RMSNorm(w)` vs `RMSNorm(1+w)` (Gemma 2/3). One flag; affects how the weight is interpreted.

3. **QK-Norm** — `{none, per-head-dim, full-flat}`. Qwen3 & Gemma3 = per-Dh; OLMo2 = full H·Dh; Llama/Mistral/Phi3 = none.

4. **Attention scale** — `{1/sqrt(Dh), 1/sqrt(qk_head_dim), config.attention_multiplier, config.query_pre_attn_scalar^-0.5, * mscale*mscale}`. One float-or-callable returns the effective scale.

5. **Position encoding** — `{rope-half-rotate (NEOX), rope-interleaved (GPT-J), rope-partial, no-rope, M-RoPE, LongRoPE, YaRN-mscale}`. RoPE config is its own object with `rope_type`, `rope_theta`, `partial_rotary_factor`, `factor`, `mscale_all_dim`, etc.

6. **QKV layout** — `{three separate, fused qkv, MLA-compressed (q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim)}`. Llama/Qwen3/Gemma3/Mistral = separate; Phi3 = fused; DeepSeek-V3 = MLA.

7. **Attention biases** — independent booleans per projection (`q_bias`, `k_bias`, `v_bias`, `o_bias`, and maybe `attention_bias` for MLA's `q_a_proj`/`kv_a_proj`/`o_proj`).

8. **Softcap** — optional `attn_logit_softcapping` (Gemma 2/3). Apply after `QK^T * scale + mask` and before softmax.

9. **Sliding window** — `{none, global, per-layer-alternating}` with a `sliding_window` size. SDPA/FlashAttention backends consume it as a kwarg.

10. **GQA / MQA / MHA** — `(H, Hk)` pair. `Hk == 1` = MQA, `Hk == H` = MHA, otherwise GQA.

11. **MLP layout** — `{separate gate/up/down, fused gate_up/down}` and activation `{silu, gelu_tanh, gelu, relu^2}`. Gate-up chunk order (`gate, up` vs `up, gate`) is a packing convention encoded by the loader.

12. **MLP biases** — `mlp_bias` boolean.

13. **Channel mixer kind** — `{dense MLP, sparse MoE, hybrid (per-layer)}`. MoE adds:
    - `n_routed_experts`, `num_experts_per_tok` (top-k)
    - Router activation: `{softmax, sigmoid}`
    - Renormalize topk weights: bool
    - Auxiliary-free correction bias: bool (DeepSeek-V3)
    - Group-limited routing: `(n_group, topk_group)`
    - Shared experts: `n_shared_experts >= 0`
    - Routed scaling: `routed_scaling_factor`
    - Expert weight layout: 3D `[E, 2*Df, D]` + `[E, D, Df]`

14. **Per-layer dense/sparse decision** — `first_k_dense_replace` (DeepSeek), `mlp_only_layers` (Qwen3-MoE), `layers_num_experts` (Jamba), `interleave_moe_layer_step` (MiniMax).

15. **Token-mixer kind** — `{attention, mamba-ssm, linear-attention, hybrid-per-layer}`. Mamba adds `mamba_d_state`, `mamba_d_conv`, `mamba_expand`, `mamba_dt_rank`.

16. **Residual scaling** — optional scalar multiplier on each residual add (Granite).

17. **Embedding scaling** — optional scalar multiplier on `embed_tokens(input_ids)` (Gemma sqrt(D), Granite, NeMo).

18. **Logits scaling** — optional `/logits_scaling` before softmax (Gemma final-softcap, Granite division).

19. **Residual dropout** — optional `resid_pdrop` (Phi-3).

20. **Cache compression** — for MLA, what gets cached: full Q/K/V (HF), or compressed `kv_lora_rank + qk_rope_head_dim` (vLLM, llama.cpp).

That gives a 20-axis parameter space. The API designer can collapse it to two layers:

- A **token-mixer trait** (attention or SSM), parametrized by axes 3–10, 15, 20.
- A **channel-mixer trait** (dense MLP or MoE), parametrized by axes 11–14.

…wired into a generic `DecoderBlock` parameterized by axes 1, 2, 16, 17, 18, 19.

---

## Surprising HF / vLLM / llama.cpp divergences

1. **MLA caching strategy**: HF caches *decompressed* K/V (size `H*qk_head_dim` + `H*v_head_dim`); vLLM and llama.cpp cache *compressed* (`kv_lora_rank + qk_rope_head_dim`) and absorb `kv_b_proj` into the attention output path. The KV cache is ~10× smaller in the production engines. This means an API that treats KV-cache as first-class must surface "absorb" as a transform separate from the math definition.

2. **Norm-and-residual fusion**: vLLM's `RMSNorm` has a `(hidden_states, residual)` two-arg overload that performs `prev_residual + hidden_states, then norm` in one kernel — a fused add-norm. HF computes them as separate ops. llama.cpp does whichever ggml has a kernel for. The graph is the same; only the kernel boundary differs.

3. **Phi-3 vLLM is one line**: `class Phi3ForCausalLM(LlamaForCausalLM)` plus a `packed_modules_mapping`. Fused QKV and fused gate-up are *the same modules* as Llama; the only thing that changes is the weight-name → module-attribute mapping. This confirms Phi-3 *is* Llama with a packing convention.

4. **QK-norm shape**: Qwen3 and Gemma3 use a `[head_dim]` gain (a single gamma broadcast across heads), but OLMo 2 uses `[H * head_dim]` (a separate gamma per head and per dim). The Qwen3 source has an explicit comment on line 248: `"unlike olmo, only on the head dim!"`. Two valid choices; the API should expose both.

5. **DeepSeek-V3 router**: uses *sigmoid* on the gate logits (not softmax) and adds a *non-learnable* `e_score_correction_bias`. The HF code path at line 215 is `router_logits = router_logits.sigmoid()` — easy to miss because Mixtral/Qwen3-MoE use `softmax`. llama.cpp encodes this as `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` vs `SOFTMAX`.

6. **Gemma3 RMSNorm `(1 + w)`**: HF computes it as `output * (1.0 + self.weight.float())` *at runtime* (line 141), with `w` initialized to zero (so gain = 1.0 at init). Some GGUF converters bake the `1+w` into the stored weights so the runtime is a plain RMSNorm. This is a known foot-gun (HuggingFace PR #29402 is referenced in a Gemma3 source comment).

7. **Llama-3 biases**: HF's `LlamaAttention` and `LlamaMLP` each have a `bias=config.attention_bias` / `bias=config.mlp_bias` flag (lines 239–249, 177–179). Llama-3 405B has *true* for attention biases on some layers. vLLM exposes this via `attention_bias` + `bias_o_proj` + `qkv_bias` (three separate config keys) and the Granite/Phi3 paths in vLLM read them differently.

8. **Mistral SDPA / FlashAttention split for SWA**: HF passes `sliding_window=...` through to the attention backend. FlashAttention 2.5+ has a native `window_size` kwarg; SDPA does not — so the SDPA path of HF falls back to a constructed sliding mask via `create_sliding_window_causal_mask`. vLLM threads the same arg through `Attention(per_layer_sliding_window=...)`. llama.cpp builds a per-layer mask with `LLAMA_SWA_TYPE_STANDARD` and a layer-pattern.

9. **Softcap costs hardware path**: Gemma3's `attn_logit_softcapping` (line 333) is *trivial* in HF eager (lines 291–294), but FlashAttention has special-case kernels (FA2 added softcap support in 2.5.7+). vLLM gates the FlashAttention backend on softcap support; otherwise it falls back to xformers / SDPA. The API must declare softcap as a fusable property so engines can pick a kernel.

10. **OLMo2 has no `input_layernorm`**: lines 295–303 of `Olmo2DecoderLayer.__init__` define `post_attention_layernorm` and `post_feedforward_layernorm` *only*. Forward sends `hidden_states` directly into `self_attn` without a pre-norm. Most readers expect every modern decoder to start with `input_layernorm`; OLMo2 breaks that assumption.

11. **Granite scaling locations**: `attention_multiplier` (replaces 1/sqrt(Dh) entirely), `residual_multiplier` (scales sublayer outputs), `embedding_multiplier` (scales the embedding table output), `logits_scaling` (divides the final logits). These are not bolted-on tricks but *load-bearing* hyperparameters from muP. llama.cpp implements them as `f_attention_scale`, `f_residual_scale`, `f_logit_scale` GGUF fields.

12. **DeepSeek-V3 dense lead**: `if layer_idx >= config.first_k_dense_replace: self.mlp = DeepseekV3MoE(config) else self.mlp = DeepseekV3MLP(config)` — at line 489–492. The first few layers are dense even in the MoE model. A vanilla MoE config that doesn't expose this fails for DeepSeek-V3.

13. **Jamba layer alternation via `n_head_kv == 0`**: llama.cpp's convention to mark a Mamba layer in a hybrid model is to set `n_head_kv[il] = 0`. HF instead uses `config.layers_block_type[il] == "mamba"`. Both work but require different config readers.

14. **Mixtral's "naive" experts loop**: HF's `MixtralExperts.forward` (lines 74–98) is a python `for expert_idx in expert_hit` loop with `index_add_`. It is correct but slow. The whole point of vLLM's `FusedMoE` and llama.cpp's `build_moe_ffn` is to replace this loop. The HF code is the *math reference*, not the production path.

15. **RoPE `interleave` flag in DeepSeek**: `apply_rotary_pos_emb_interleave` (lines 320–355) reshapes Q/K so that an interleaved (NORM/GPT-J) layout can be reused with the half-rotate cos/sin tables. This is a workaround for weight-format compatibility, not a math change.

---

## Things I couldn't access cleanly

- The `src/llama-graph.cpp` `build_attn` overloads I read via WebFetch returned only summaries (the file is large enough that the fetched HTML truncated the relevant blocks); exact line ranges would need a local clone of llama.cpp.
- Some vLLM files (e.g. `deepseek_v3.py`) returned 404 — vLLM ships the MLA in `deepseek_v2.py` and DeepSeek-V3 derives from it via the V3 config. The MoE divergences (sigmoid routing, e_score_correction_bias, group routing) are realized through `FusedMoE` kwargs and are documented above from the v2 file.
- I did not pull the Phi-3 LongRoPE short-vs-long factor table or the Gemma3 vision tower (out of scope for the decoder block).
- llama.cpp Mistral / Mixtral graph files (`mistral4.cpp`, `mixtral.cpp`) returned only partial summaries; the per-layer code reuses generic LLM_ARCH_LLAMA/LLM_ARCH_LLAMA-MoE paths plus a small SWA hook, which is reflected in the report.
- The HF transformers repo CLAUDE.md (referenced via a system reminder) points to `.ai/AGENTS.md`; I did not retrieve that file because it is outside the scope of this layer-source survey.

All file paths and line numbers in this report were checked against the local `transformers` clone at `C:\Users\zhengte\external\transformers\src\transformers\models\` so the citations can be verified directly. vLLM and llama.cpp pseudocode reflects the current `main` / `master` of each repo as of the WebFetch retrieval.

---

## Appendix A — A unified pseudocode of "the decoder block", parametrized

Putting all 20 axes together, the universal block can be written as a single Python function. This is the shape an API floor should support; everything below is dispatched by configuration.

```python
def decoder_block(
    h, residual,                              # [B, S, D], [B, S, D]
    *,
    layer_idx, position_embeddings, attention_mask, past_key_values,
    # token mixer
    token_mixer_kind,         # "attention" | "mamba" | "linear-attn"
    token_mixer_params,       # struct of attention or SSM params
    # channel mixer
    channel_mixer_kind,       # "mlp" | "moe"
    channel_mixer_params,     # struct of MLP or MoE params
    # norm placement
    norm_kind,                # "rms" | "rms_one_plus" | "layer"
    norm_placement,           # "pre" | "post" | "sandwich"
    eps,
    # scaling
    residual_multiplier=1.0,  # Granite; default 1
    # dropout
    resid_pdrop=0.0,          # Phi-3
):
    # --- Token mixer ---
    if norm_placement in ("pre", "sandwich"):
        x = norm(h, eps, kind=norm_kind, weight=W_attn_pre)
    else:
        x = h
    y = token_mixer(x, position_embeddings, attention_mask,
                    past_key_values, layer_idx, **token_mixer_params)
    if norm_placement in ("post", "sandwich"):
        y = norm(y, eps, kind=norm_kind, weight=W_attn_post)
    if resid_pdrop > 0: y = dropout(y, resid_pdrop)
    h = residual_add(h, y, scale=residual_multiplier)

    # --- Channel mixer ---
    if norm_placement in ("pre", "sandwich"):
        x = norm(h, eps, kind=norm_kind, weight=W_ffn_pre)
    else:
        x = h
    y = channel_mixer(x, **channel_mixer_params)
    if norm_placement in ("post", "sandwich"):
        y = norm(y, eps, kind=norm_kind, weight=W_ffn_post)
    if resid_pdrop > 0: y = dropout(y, resid_pdrop)
    h = residual_add(h, y, scale=residual_multiplier)
    return h
```

`token_mixer(...)` for attention:

```python
def attention(x, position_embeddings, attention_mask, kv_cache, layer_idx, *,
              num_heads, num_kv_heads, head_dim,
              qkv_layout,           # "split" | "fused" | "mla"
              qk_norm_shape,        # None | "head_dim" | "full"
              rope_kind,            # "neox" | "norm" | "partial" | "no_rope" | "mrope"
              partial_rotary_factor,
              sliding_window,       # None | int
              attn_scale,           # 1/sqrt(Dh) by default, override per-family
              attn_logit_softcap,   # None | float
              biases):              # dict of bool per projection
    # 1. Project
    if qkv_layout == "fused":
        qkv = qkv_proj(x); q, k, v = qkv.split([H*Dh, Hk*Dh, Hk*Dh], -1)
    elif qkv_layout == "split":
        q, k, v = q_proj(x), k_proj(x), v_proj(x)
    elif qkv_layout == "mla":
        q = q_b_proj(rms_norm(q_a_proj(x), eps))
        compressed_kv = kv_a_proj_with_mqa(x)
        k_pass, k_rot = compressed_kv.split([kv_lora_rank, qk_rope_head_dim], -1)
        k_pass = kv_b_proj(rms_norm(k_pass, eps))
        k_pass, v = k_pass.split([qk_nope_head_dim, v_head_dim], -1)
    # 2. Reshape into heads
    q = q.view(B, S, H, head_dim).transpose(1, 2)        # [B, H, S, Dh]
    k = k.view(B, S, Hk, head_dim).transpose(1, 2)
    v = v.view(B, S, Hk, v_head_dim_or_Dh).transpose(1, 2)
    # 3. QK-norm (optional)
    if qk_norm_shape == "head_dim":
        q = rms_norm_per_head(q, eps)
        k = rms_norm_per_head(k, eps)
    elif qk_norm_shape == "full":
        q = rms_norm_flat(q, eps)
        k = rms_norm_flat(k, eps)
    # 4. Position encoding
    if rope_kind == "partial":
        q, k = apply_partial_rope(q, k, position_embeddings, partial_rotary_factor)
    elif rope_kind in ("neox", "norm"):
        q, k = apply_rope(q, k, position_embeddings, kind=rope_kind)
    # MLA-specific: cat rope and nope halves after rope on rope-half only
    # 5. Cache update — either compressed or decompressed
    k, v = kv_cache.update(k, v, layer_idx)
    # 6. Scaled dot-product or flash
    attn = scaled_dot_product_attention(
        q, k, v, mask=attention_mask, scale=attn_scale,
        softcap=attn_logit_softcap, sliding_window=sliding_window)
    # 7. Output projection
    return o_proj(attn.transpose(1, 2).reshape(B, S, H * v_head_dim))
```

`channel_mixer(...)` for MLP / MoE:

```python
def mlp_swiglu(x, *, gate_proj, up_proj, down_proj, act):
    return down_proj(act(gate_proj(x)) * up_proj(x))

def mlp_fused_gate_up(x, *, gate_up_proj, down_proj, act, chunk_order):
    gu = gate_up_proj(x)
    if chunk_order == "gate_first":
        gate, up = gu.chunk(2, -1)
    else:
        up, gate = gu.chunk(2, -1)
    return down_proj(act(gate) * up)

def moe_block(x, *, router, experts, shared_experts,
              router_activation,            # softmax | sigmoid
              top_k, renormalize, score_correction_bias,
              n_group, topk_group, routed_scaling):
    logits = router(x)
    if router_activation == "softmax":
        probs = logits.softmax(-1)
    else:
        probs = logits.sigmoid()
    if score_correction_bias is not None:
        choice_scores = probs + score_correction_bias
    else:
        choice_scores = probs
    if n_group:                    # group-limited
        mask = group_topk_mask(choice_scores, n_group, topk_group)
        choice_scores = choice_scores.masked_fill(~mask, -inf)
    top_w, top_idx = choice_scores.topk(top_k, -1)
    if renormalize:
        top_w = top_w / top_w.sum(-1, keepdim=True)
    top_w = top_w * routed_scaling
    routed = dispatch_experts(x, experts, top_idx, top_w)
    if shared_experts is not None:
        routed = routed + shared_experts(x)
    return routed
```

`token_mixer(...)` for Mamba SSM:

```python
def mamba_mixer(x, ssm_state_cache, *, in_proj, conv1d, x_proj, dt_proj,
                A_log, D, out_proj, dt_rank, ssm_state_size):
    proj = in_proj(x).transpose(1, 2)            # [B, 2*E_int, S]
    x_inner, gate = proj.chunk(2, dim=1)
    x_inner = causal_conv1d(x_inner, conv1d.weight, conv1d.bias, activation="silu")
    ssm_params = x_proj(x_inner.transpose(1, 2))
    dt, B_, C_ = ssm_params.split([dt_rank, ssm_state_size, ssm_state_size], dim=-1)
    dt = dt_proj(dt).transpose(1, 2)
    A = -A_log.float().exp()
    y = selective_scan(x_inner, dt, A, B_, C_, D, gate, ssm_state_cache)
    return out_proj(y.transpose(1, 2))
```

These three primitives (attention, mamba, swiglu/moe) + the `decoder_block` shell are sufficient to compose every family in this report.

---

## Appendix B — A side-by-side norm-placement table

| Family            | Pre-attn norm | Post-attn norm | Pre-ffn norm | Post-ffn norm | Output norm |
|-------------------|---------------|----------------|--------------|---------------|-------------|
| Llama 3           | input_layernorm | —            | post_attention_layernorm | — | model.norm |
| Qwen 3            | input_layernorm | —            | post_attention_layernorm | — | model.norm |
| Mistral           | input_layernorm | —            | post_attention_layernorm | — | model.norm |
| Phi-3             | input_layernorm | —            | post_attention_layernorm | — | model.norm |
| Granite           | input_layernorm | —            | post_attention_layernorm | — | model.norm |
| Mixtral / Qwen3-MoE | input_layernorm | —          | post_attention_layernorm | — | model.norm |
| DeepSeek-V3       | input_layernorm | —            | post_attention_layernorm | — | model.norm |
| Jamba             | input_layernorm | —            | pre_ff_layernorm | —         | model.final_layernorm |
| **OLMo 2**        | **—**         | **post_attention_layernorm** | — | **post_feedforward_layernorm** | model.norm |
| **Gemma 3**       | input_layernorm | **post_attention_layernorm** | **pre_feedforward_layernorm** | **post_feedforward_layernorm** | model.norm |

The structural majority is pre-norm; OLMo 2 is the post-norm outlier; Gemma 3 is the sandwich-norm outlier. The downstream API enum should be `{PRE, POST, SANDWICH}`, applied independently to the attention and FFN sublayers.

---

## Appendix C — QKV layout sub-axis

| Family            | Layout in HF             | Layout in vLLM                | Layout in llama.cpp                |
|-------------------|--------------------------|-------------------------------|-------------------------------------|
| Llama 3           | 3 × `Linear`             | `QKVParallelLinear` (fused)   | `wq`, `wk`, `wv` (or fused `wqkv`)  |
| Qwen 3            | 3 × `Linear` + `q_norm`/`k_norm` | `QKVParallelLinear` + per-head norm | `wq`, `wk`, `wv` + `attn_q_norm`, `attn_k_norm` |
| Gemma 3           | 3 × `Linear` + `q_norm`/`k_norm` | Same as Qwen3            | Same |
| **Phi-3**         | `qkv_proj` fused         | Inherits Llama's `QKVParallelLinear` | `wq`/`wk`/`wv` separate (loaded), or fused `wqkv` |
| Mistral           | 3 × `Linear`             | `QKVParallelLinear`           | `wq`, `wk`, `wv` |
| **DeepSeek-V3**   | `q_a_proj` → `q_a_norm` → `q_b_proj`; `kv_a_proj_with_mqa` → `kv_a_norm` → `kv_b_proj` | `fused_qkv_a_proj` + MLA absorb | `wq_a`, `wq_a_norm`, `wq_b`, `wkv_a_mqa`, `attn_kv_a_norm`, `wkv_b` |
| Mixtral           | 3 × `Linear`             | `QKVParallelLinear`           | `wq`, `wk`, `wv` |
| Granite           | 3 × `Linear` (+ optional biases) | `QKVParallelLinear`     | `wq`, `wk`, `wv` (+ optional bias tensors) |
| OLMo 2            | 3 × `Linear` + flat `q_norm`/`k_norm` | `QKVParallelLinear` + flat norm | `wq`, `wk`, `wv` + `attn_q_norm`, `attn_k_norm` |
| Jamba (attn layer)| 3 × `Linear`             | `QKVParallelLinear`           | `wq`, `wk`, `wv` |

The API's loader must therefore accept three input formats per family and emit a canonical internal representation. The cleanest internal form is "logical" Q/K/V slots (each a tensor of shape `[D_in, n_heads*head_dim]`), with a per-family "merge spec" describing how a single GGUF/safetensor file maps onto them.

---

## Appendix D — Where biases hide

Bias presence is a per-projection bool, not a single model-level flag. Concrete counts per family (✓ = bias possible, ✗ = always false):

| Family       | Q-bias | K-bias | V-bias | O-bias | MLP-bias | Router bias | LM head |
|--------------|--------|--------|--------|--------|----------|-------------|---------|
| Llama 3 8B   |   ✗    |   ✗    |   ✗    |   ✗    |    ✗     |     —       |    ✗    |
| Llama 3 405B |   ✓    |   ✓    |   ✓    |   ✗    |    ✗     |     —       |    ✗    |
| Qwen 3       |   ✓    |   ✓    |   ✓    |   ✗    |    ✗     |     —       |    ✗    |
| Qwen 2       |   ✓    |   ✓    |   ✓    |   ✗    |    ✗     |     —       |    ✗    |
| Gemma 3      |   ✗    |   ✗    |   ✗    |   ✗    |    ✗     |     —       |    ✗    |
| Phi-3        |   ✗    |   ✗    |   ✗    |   ✗    |    ✗     |     —       |    ✗    |
| Mistral      |   ✗    |   ✗    |   ✗    |   ✗    |    ✗     |     —       |    ✗    |
| DeepSeek-V3  |   ✓ (on q_a_proj only) | ✓ (on kv_a_proj_with_mqa) | (no v_proj) | ✓ | ✗ | DeepSeek's `e_score_correction_bias` (non-trainable buffer) | ✗ |
| Mixtral      |   ✗    |   ✗    |   ✗    |   ✗    |    ✗     | ✗ (router weight only) | ✗ |
| Granite      |   ✓    |   ✓    |   ✓    |   ✓    |    ✓     |     —       |    ✗    |
| OLMo 2       |   ✗    |   ✗    |   ✗    |   ✗    |    ✗     |     —       |    ✗    |
| Jamba        |   ✗    |   ✗    |   ✗    |   ✗    |    ✗     |     —       |    ✗    |

The `Llama 3 405B` row is a real gotcha: most code assumes `attention_bias=False` for all Llama variants. The HF config carries `attention_bias` per checkpoint; vLLM has `attention_bias`, `bias_o_proj`, and `qkv_bias` (three separate config keys, lines 220–230 of vLLM `llama.py`); llama.cpp has independent `wq_b`, `wk_b`, `wv_b`, `wo_b` tensor slots and treats `nullptr` as no-bias. The API needs four independent bias flags for the attention projections.

---

## Appendix E — Scale and constant factors per family (numeric)

Sources: each family's HF config defaults (representative checkpoints).

| Family        | head_dim | attention scale          | rope_theta | rms_norm_eps | embed_scale | residual_scale | logits_scale | softcap |
|---------------|----------|---------------------------|------------|--------------|-------------|----------------|--------------|---------|
| Llama-3-8B    | 128      | 1/sqrt(128) ≈ 0.0884     | 5e5        | 1e-5         | 1.0         | 1.0            | 1.0          | none    |
| Qwen3-8B      | 128      | 1/sqrt(128)              | 1e7        | 1e-6         | 1.0         | 1.0            | 1.0          | none    |
| Gemma-3-4B    | 256      | 1/sqrt(query_pre_attn_scalar=256) = 1/16 = 0.0625 | 1e6 (global) / 1e4 (sliding) | 1e-6 | sqrt(2560)≈50.6 | 1.0 | softcap=30.0 | attn_softcap=50.0 |
| Phi-3-Mini    | 96       | 1/sqrt(96)               | 1e4 (short) / dynamic-rope (long) | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| Mistral-7B    | 128      | 1/sqrt(128)              | 1e6        | 1e-5         | 1.0         | 1.0            | 1.0          | none    |
| DeepSeek-V3   | 192 (qk_head_dim) | 1/sqrt(192) * mscale^2 | YaRN-extended (10000 * factor) | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| Mixtral-8x7B  | 128      | 1/sqrt(128)              | 1e6        | 1e-5         | 1.0         | 1.0            | 1.0          | none    |
| Granite-3.1-3B| 64       | 0.0078125 (=attn_mult)    | 5e6        | 1e-5         | 12.0        | 0.22           | 8.0          | none    |
| OLMo-2-7B     | 128      | 1/sqrt(128)              | 5e5        | 1e-6         | 1.0         | 1.0            | 1.0          | none    |
| Jamba-52B     | 128      | 1/sqrt(128)              | 1e4        | 1e-6         | 1.0         | 1.0            | 1.0          | none    |

Notable: Granite is the only family where every column except `rope_theta` is non-default. Gemma 3 is the only family with softcap. DeepSeek-V3 is the only one with a YaRN mscale^2 multiplier baked into the attention scale.

---

## Appendix F — Citations cross-reference for the API designer

Direct file/line citations to verify any claim above:

- Llama 3 HF: `transformers/src/transformers/models/llama/modeling_llama.py` lines 52–332
- Qwen 3 HF: `transformers/src/transformers/models/qwen3/modeling_qwen3.py` lines 49–334
- Gemma 3 HF: `transformers/src/transformers/models/gemma3/modeling_gemma3.py` lines 98–428
- Phi-3 HF: `transformers/src/transformers/models/phi3/modeling_phi3.py` lines 49–335
- Mistral HF: `transformers/src/transformers/models/mistral/modeling_mistral.py` lines 35–240
- DeepSeek-V3 HF: `transformers/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py` lines 37–526
- Mixtral HF: `transformers/src/transformers/models/mixtral/modeling_mixtral.py` lines 61–389
- Granite HF: `transformers/src/transformers/models/granite/modeling_granite.py` lines 115–504
- OLMo 2 HF: `transformers/src/transformers/models/olmo2/modeling_olmo2.py` lines 51–333
- Jamba HF: `transformers/src/transformers/models/jamba/modeling_jamba.py` lines 148–638

- Llama vLLM: `vllm/model_executor/models/llama.py` — `LlamaDecoderLayer`, `LlamaAttention`, `LlamaMLP`
- Qwen3 vLLM: `vllm/model_executor/models/qwen3.py`
- Gemma3 vLLM: `vllm/model_executor/models/gemma3.py`
- Phi-3 vLLM: `vllm/model_executor/models/phi3.py` (one-line inheritance from Llama)
- Mistral vLLM: `vllm/model_executor/models/mistral.py`
- DeepSeek-V2/V3 vLLM: `vllm/model_executor/models/deepseek_v2.py` (`DeepseekV2MLAAttention`, `DeepseekV2MoE`)
- Mixtral vLLM: `vllm/model_executor/models/mixtral.py`
- Granite vLLM: `vllm/model_executor/models/granite.py`
- OLMo 2 vLLM: `vllm/model_executor/models/olmo2.py`
- Jamba vLLM: `vllm/model_executor/models/jamba.py`

- Llama llama.cpp: `src/models/llama.cpp` + `src/llama-graph.cpp::build_attn`, `build_ffn`, `build_norm`
- Qwen 3 llama.cpp: `src/models/qwen3.cpp`
- Gemma 3 llama.cpp: `src/models/gemma3.cpp` (also `gemma.cpp`, `gemma2.cpp` for older variants)
- Phi-3 llama.cpp: `src/models/phi3.cpp` (also `phi2.cpp`)
- Mistral llama.cpp: `src/models/mistral4.cpp` and the generic LLM_ARCH_LLAMA graph
- DeepSeek-V3 llama.cpp: `src/models/deepseek2.cpp` (V3 reuses V2's graph)
- Mixtral llama.cpp: generic LLM_ARCH_LLAMA-MoE path (no dedicated mixtral.cpp; uses build_moe_ffn)
- Granite llama.cpp: `src/models/granite.cpp` (and `granite-moe.cpp`)
- OLMo 2 llama.cpp: `src/models/olmo2.cpp` (note `olmo.cpp` is OLMo 1)
- Jamba llama.cpp: `src/models/jamba.cpp` (with `build_mamba_layer` from `src/llama-graph.cpp`)

These ten triples (HF × vLLM × llama.cpp) form the verifiable test set for any candidate API. A successful API converts any one HF `<Family>DecoderLayer.forward` into an equivalent ggml graph or vLLM module by *configuration alone* — no per-family code.
