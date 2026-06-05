# Critique of `02-layer-sources.md` — Per-Layer Source Survey

**Reviewer stance:** Skeptical / harsh. The current document is a competent *engineering note* for the ten Llama-adjacent families the author already knew. It is **not yet a survey paper** of "LLM-layer evolution over three years" because (a) it omits whole architectural lineages that are *the* counterexamples to the Llama paradigm, (b) several pseudocode blocks are subtly mis-shaped or mis-ordered, and (c) the divergence story between HF, vLLM, and llama.cpp is one-sided in favor of HF.

This critique is organized into six sections per the brief: missing families, pseudocode quality, divergence completeness, evolution narrative, correctness errors, and recommended fixes for v2.

---

## Section 1 — Missing model families (table)

The document promises "ten SLM families" and delivers exactly ten — but the families chosen are all decoder-only post-norm-or-pre-norm SwiGLU transformer derivatives. The *interesting* architectural deltas of the last three years happen *outside* this set. The table below lists what must be added for the report to deserve the title "survey of LLM layers".

Severity legend: **CRIT** = the family introduces a new axis not currently in the 20-axis taxonomy; **HIGH** = the family introduces a *quantitatively important* variant on an existing axis; **MED** = the family is a popular checkpoint whose absence is mostly editorial.

| Family | Why architecturally distinct | Severity |
|---|---|---|
| **BitNet b1.58** | (a) Ternary {-1,0,+1} weights with per-layer scale, totally different arithmetic from fp16/bf16. (b) Introduces TWO sub-norms inside the block: `attn_sub_norm` between attention and `o_proj` (line 216 of `modeling_bitnet.py`) AND `ffn_sub_norm` between `act(gate)*up` and `down_proj` (line 74). These are NOT in the report's 20-axis space and are not mentioned anywhere. (c) Reference `modeling_bitnet.py` exists in the local clone. | **CRIT** |
| **OpenELM** | Per-layer head/FFN scaling (DeepNet-style depth-decoupled width). Each transformer block has a *different* `num_heads`, `head_dim`, and `ffn_multiplier`. This breaks the "single config" assumption of all ten families currently surveyed and the API-floor design in Appendix A. | **CRIT** |
| **Mamba-2** | Pure SSM with state-space duality (matrix form). The reference `modeling_mamba2.py` exists in the local clone. Jamba's SSM block is closer to Mamba-1; Mamba-2 introduces a different parameterization (`A` is now a scalar per channel, GroupNorm before C, structured-mask attention duality). Reporting Jamba's mixer as "the SSM reference" is misleading without distinguishing M1 vs M2. | **CRIT** |
| **RecurrentGemma (Griffin)** | Gated linear recurrent unit (RG-LRU) + local attention, completely separate recurrence math from Mamba. Already in HF as `recurrent_gemma`. The report's claim that Jamba is "the non-pure-transformer reference" is wrong — Griffin is the *other* major non-pure-transformer line. | **CRIT** |
| **Falcon-Mamba / Falcon-H1** | Pure Mamba (Falcon-Mamba) and hybrid (Falcon-H1). HF directories `falcon_mamba` and `falcon_h1` exist. Falcon-H1 specifically uses parallel Mamba+attention, a different topology from Jamba's serial alternation. | **CRIT** |
| **Zamba / Zamba2** | Mamba2 backbone + a *shared* attention block reused across layers (parameter sharing — a new axis). Local clones `zamba/`, `zamba2/`. The report's per-layer parameter assumption breaks here. | **CRIT** |
| **Granite-MoE-Hybrid** (`granitemoehybrid`) | Local directory confirms a Granite × Jamba hybrid lineage; not even mentioned, despite Granite being section 8. | HIGH |
| **OLMoE** | Sparse MoE OLMo with the OLMo2 post-norm layout but Mixtral-style softmax routing — a useful corner of (norm-placement) × (MoE-router) that's not covered. Directory `olmoe/`. | HIGH |
| **Nemotron / Nemotron-H** | Nemotron-H is NVIDIA's *production* Mamba-attention hybrid (different from Jamba's 1:7 ratio). Nemotron-4 also uses squared-ReLU activation in MLP — the report's activation enum `{silu, gelu_tanh, gelu, relu^2}` mentions `relu^2` but cites no family for it. Nemotron is the citation. | HIGH |
| **Cohere Command-R / Aya** | Tied embeddings + scaled logits (`logit_scale`, similar idea to Granite but with a different formula), no biases, and the Cohere2 variant uses sliding-window alternation similar to Gemma 3 but with a different period. Directories `cohere/`, `cohere2/`. | HIGH |
| **Cohere2-MoE** | Recent MoE checkpoint with `cohere2_moe/` directory; differs from Mixtral on routing. | HIGH |
| **Hunyuan-V1 (dense + MoE)** | Tencent's recent open-weight family, directory `hunyuan_v1_dense/` and `hunyuan_v1_moe/`. Uses a cross-layer shared MoE configuration the report does not classify. | HIGH |
| **MiniMax / MiniMax-M2** | The MiniMax family uses *Lightning Attention* (a linear-attention variant) in alternation with softmax attention — the report's axis 15 names "linear-attn" but no family is enumerated. Directories `minimax/`, `minimax_m2/`. | HIGH |
| **GPT-OSS** | OpenAI's open-weight checkpoint, directory `gpt_oss/`. Uses sink tokens + sliding-window with a specific layout. Architectural details not covered. | HIGH |
| **Exaone-4 / Exaone-MoE** | LG's recent family, directories present. Differs on RoPE base + QK clipping. | MED |
| **Phi-3-small** | Uses *blocksparse* attention (different from Phi-3-mini's dense). The report covers Phi-3-mini's quirks (fused QKV, partial RoPE, residual dropouts) but never mentions Phi-3-small or its blocksparse pattern. | HIGH |
| **Phi-4-mini-flash / Samba** | Microsoft's recent SLM uses the *Samba* architecture (Mamba + sliding-window attention + MLP, NOT the Jamba alternation pattern). Genuinely new topology. | HIGH |
| **Llama 3.2 1B / 3B** | Edge-targeted: embedding-LM-head sharing (different from Llama 3.1 8B), quantization-aware training, knowledge-distilled. Not architecturally radical but the *embedding-sharing flag* belongs on the bias/scale table in Appendix D. | MED |
| **SmolLM2 / SmolLM3** | `smollm3/` directory exists. Decoupled embed/lm_head, tied-weight choice differences, lower `rms_norm_eps`. SmolLM3 also adds NoPE alternation (some layers literally have *no* position encoding) — a new value on axis 5. | HIGH |
| **DeepSeek-V2** (separate from V3) | The *original* MLA introduction. V2 uses softmax routing without bias correction; V3 added the no-aux bias. Conflating them hides where each idea entered the literature. Directory `deepseek_v2/` exists separately from `deepseek_v3/`. | HIGH |
| **DeepSeek-V4** | Local clone has `deepseek_v4/`. Not mentioned. | MED |
| **Qwen3-Next / Qwen3.5** | `qwen3_next/`, `qwen3_5/`, `qwen3_5_moe/` directories. Qwen3-Next alternates *gated DeltaNet* (linear attention) with softmax attention. Mentioned in passing ("Qwen3-Next further adds the linear-attention alternation, which we don't enumerate here") — but a survey CANNOT punt on this. | **CRIT** |
| **InternLM2 / InternLM3** | Dynamic NTK RoPE scaling (a different `rope_type` than YaRN/LongRoPE). Mentioned nowhere. | MED |
| **Yi-1.5** | Cosine-similarity attention variant in early Yi (later dropped). A "dead end" worth noting per the evolution-narrative criterion. | MED |
| **GLM-4 9B** | RoPE half-truncation + prefix LM heritage. `chatglm` family covered separately in HF. | MED |
| **Baichuan 1/2** | ALiBi in Baichuan-1 (7B) → RoPE in Baichuan-2. A concrete "ALiBi → RoPE" data point that's literally absent from the evolution narrative. | MED |
| **ChatGLM 1/2/3** | RoPE + prefix LM; uses RMSNorm in some variants and LayerNorm in others. Useful early data point. | MED |
| **Falcon-7B / Falcon-40B** | Multi-query attention (`Hk = 1`) and *parallel* attn+FFN (`x + attn(norm(x)) + mlp(norm(x))` — parallel residual). The report's axis 1 (norm placement) has *no value* for parallel-residual, yet it claims to be a universal API floor. | **CRIT** |
| **StarCoder 2** | Grouped-query attention with FIM (fill-in-middle) support; uses 2-norm layer with a special sliding-window pattern; some variants are MoE. Code-LLM lineage entirely absent. | HIGH |
| **RWKV-6 / RWKV-7** | Pure recurrent neural network with WKV (weighted-key-value) operator. Not transformer at all. If the survey wants to be honest about "LLM layers in the past 3 years", RWKV-7 (2024) is mandatory. Directory `rwkv/` exists. | **CRIT** |
| **Mistral-Large / Pixtral** | Mistral-Large uses position-dependent attention scaling (the `1 + beta*log(1 + floor(pos/orig_max))` formula that the report mentions in passing under vLLM Mistral). This is a quantitatively important divergence and deserves its own subsection. | MED |
| **Apple Foundation Model** | Documented in Apple's WWDC 2024 paper (LoRA adapters on-device + quantization). Out of HF, but the survey claims to cover "on-device" SLMs and Apple is the most-deployed on-device LLM in the world. | MED |
| **Persimmon / StableLM / Arcee / Aria** | Smaller but distinct lineages whose configs would tighten the report's empirical coverage. | MED |

**Count of missing families: ~30 (15 CRIT/HIGH + 15 MED).** The current document covers 10. The survey-paper bar is closer to 30–40.

---

## Section 2 — Pseudocode quality issues per family

### 2.1 Llama 3 (section 1)

- The pseudocode on lines 75–84 writes `attn = softmax((q @ k.T) * head_dim**-0.5 + mask) @ v` but elides the `transpose(2,3)` and the fact that mask is *added* in fp32 (line 211 in HF source). For a survey paper, the dtype-promotion-and-cast story matters and is missing.
- Line 82 in target md: `q @ k.T` is ambiguous — is it the full transpose or only the last two dims? Source uses `transpose(2,3)`. The pseudocode should match.
- The `repeat_kv` step is described as "→ [B, H, S, Dh] for eager path; SDPA/flash do it internally" but the actual repeat is `expand` along a *new* dim then `reshape` — a memory-non-allocating broadcast. This matters for the API design (especially for vLLM's tensor-parallel split).
- Bias claim on line 96 ("Llama-3.x 405B uses biases on some attention projections") is sourced to HF lines 239–249 but the actual fact is the *config flag* `attention_bias` defaults to False; whether 405B uses True is *deployment-checkpoint-dependent*. The claim is plausible but uncited. Footnote needed.

### 2.2 Qwen 3 (section 2)

- **Correct on the QK-norm placement** (verified: target md line 263 = HF source line 263 in `modeling_qwen3.py`). Norm sits between view-into-heads and transpose-to-`[B,H,S,Dh]`, before RoPE.
- However, the report does **not** flag the asymmetry with Gemma 3: in Gemma 3 (HF lines 351–356) the q_norm runs *after* `transpose(1,2)`, so the input shape at norm time is `[B,H,S,Dh]` (norms on last-dim Dh, same math). In Qwen 3, norm sits BEFORE transpose, so shape is `[B,S,H,Dh]` (still last-dim Dh). The math is identical but for kernel-fusion purposes the two cases compile to different kernels. The report's universal-API pseudocode in Appendix A elides this; a survey should at least name the divergence.
- The vLLM block (target md lines 246–256) writes `q_by_head = q.view(-1, num_heads_local, head_dim)` — this is a *flat-token* (2D) layout used by `QKVParallelLinear`, which is correct for vLLM but the *shape annotation* `[-1, H, Dh]` is incomplete: vLLM batches across `(B*S)`, so the actual leading dim is "tokens", not "B*S" when continuous batching is in use. The annotation should be `[T, H, Dh]` with a note that `T = sum(seq_len for each request in batch)`.
- The sliding-window claim on target md line 233 — "Qwen3-8B currently uses full attention only" — is unverified. Qwen3 *does* support per-layer SWA via `config.layer_types`; whether the 8B *checkpoint config* sets it is a release-specific fact. Without a config-file citation this is hand-waving.

### 2.3 Gemma 3 (section 3)

- The report's "sandwich norm" label is *defensible* but imprecise. The actual structure (HF lines 407–426) is:
  ```
  out = x + post_attn_norm(attn(input_layernorm(x)))
  out = out + post_ff_norm(mlp(pre_ff_norm(out)))
  ```
  This is NOT pre-norm-and-post-norm-around-the-same-residual (the original "sandwich norm" of CogView/Cogview2). It's pre-norm-PLUS-an-extra-norm-on-the-sublayer-output. Calling it "sandwich" is a community convention but a survey should disambiguate.
- The attention scaling claim on target md line 336 is correct: `query_pre_attn_scalar**-0.5` (verified at HF line 317). However, the report doesn't explain *why* Gemma uses `query_pre_attn_scalar` as a configurable knob — it lets the team decouple `head_dim` from the attention scale, e.g. set `head_dim=256` but scale by `1/sqrt(256)` or by `1/sqrt(1)` for ablations. Worth one sentence.
- Softcap pseudocode (target md lines 351–355) is structurally correct but misses the dtype: HF computes the softcap in the same dtype as `attn_weights` (fp32 after the `* scaling`), then casts back when assigning to attn_output. Off-by-cast bugs are real in fp16 inference.
- The RMSNorm `(1 + w)` story on target md lines 357–366 is correct AND correctly cites HF lines 137–142. But the citation to "llama.cpp #29402 discussion" actually points to a *transformers* PR, not a llama.cpp PR — minor mistake but it's in a published file.
- The MLP activation: target md line 408 says "Gemma uses GELU" — verified (HF line 121: `self.act_fn = ACT2FN[config.hidden_activation]` with config default `"gelu_pytorch_tanh"`). The report should call out that this is the *exact-gelu* (tanh-approximation) variant, not `gelu_new` or `gelu_fast` — three different functions in `ACT2FN`.

### 2.4 Phi-3 (section 4)

- Fused gate-up chunk-order claim (target md line 440): "gate is the **first** half, up is the **second** half". Verified at HF lines 56–58 of `modeling_phi3.py`. But the report fails to mention that *Llama's* fused `gate_up_proj` (in vLLM's `MergedColumnParallelLinear`) uses the *opposite* order in some packings — this is the *exact* footgun the survey should highlight, and it's only mentioned in Appendix A as a `chunk_order` parameter.
- Partial RoPE block (target md lines 462–469): pseudocode is correct. But the report doesn't explain *why* partial RoPE was introduced (it's a GPT-NeoX inheritance; partial rotary was discovered to be comparable to full rotary at lower FLOPs in the original Phi paper). A survey paper needs the *citation* and *motivation*.
- Residual-dropout block (target md lines 474–481): correct, but the pseudocode swap "diff w/ Llama" is misleading — Llama-2 and Llama-3 *also* have a `residual_dropout` config option that defaults to 0; the difference is Phi-3 has it explicitly in the layer code path while Llama leaves it as a config that's never used. So this is a code-style divergence, not an architectural one.

### 2.5 Mistral (section 5)

- The position-dependent scaling formula on target md lines 530–534 is **mis-attributed**. The `1 + beta * log(1 + floor(positions / original_max_position_embeddings))` formula is from **Mistral-Large 2 / Llama-4**, not Mistral-7B / Mixtral-8x7B. The report says "this falls back to no scaling" for canonical Mistral-7B — that's correct — but the framing implies the formula belongs to the Mistral family at large. It doesn't; it's a Llama-4 invention also picked up by Mistral-Large 2. The chronology is wrong.
- The `freq_base = 1e6` (Mistral) vs `1e7` (some checkpoints) claim on target md line 540 is sourced to "vs Llama's 5e5/1e6". The 5e5 is **Llama-2's** value; Llama-3 uses 5e5 for 8B but switches to 5e5 *also* for 70B with a different ntk scale. Without precise per-checkpoint values, this comparison is too vague for a survey.

### 2.6 DeepSeek-V3 (section 6)

- The MLA shapes in the `__init__` block (target md lines 575–595) are correct against HF source (verified at lines 383–414). However, the pseudocode names some variables (`q_states`, `compressed_kv`) that don't match HF's variable names exactly. For a survey, the variable names should match HF source to make spot-checking easy.
- The `k_rot.view(B, 1, S, qk_rope_head_dim)` claim on target md line 611 is verified at HF line 441. The "shared key-rope across heads" annotation is correct and important. Good.
- The MoE pseudocode (target md lines 656–693) is mostly correct but the *group routing* step has a subtle bug in the report's pseudocode: line 671 says `view(-1, n_group, n_routed_experts // n_group)` but it should be `view(B*S, n_group, n_routed_experts // n_group)` — the `-1` resolves to `B*S` only if the input is a flat 2D tensor; if it's still `[B, S, E]` the `view` either fails or gives the wrong shape. The HF source flattens *before* this step (lines around 224–229 in source); the report's pseudocode should make that flatten explicit.
- Sigmoid-not-softmax claim verified at HF line 215.
- The `routed_scaling_factor` multiplier is applied *after* `norm_topk_prob` renormalization. The report shows this order correctly but doesn't flag that this means the renormalized weights *cease to sum to 1*. That's deliberate (DeepSeek-V3 wants scaled contributions) but worth noting because it breaks the "MoE outputs are a convex combination" mental model.

### 2.7 Mixtral / Qwen3-MoE (section 7)

- Mixtral attention "is identical to Mistral, including SWA" — verified, but the report should note Mixtral 8x7B's *config* sets `sliding_window=None` despite the model class supporting it. Hidden footgun: loading a Mistral-7B checkpoint into Mixtral or vice versa requires reading `sliding_window` from the config, not the class.
- The `MixtralExperts` 3D weight layout `[E, 2*Df, D]` (target md line 780) is verified, but the report should mention that vLLM's `FusedMoE` *transposes* this to `[E, D, 2*Df]` for better-stride GEMM — a real on-disk vs in-kernel divergence.
- The naive Python loop critique (target md line 783) is good but should be benchmarked. A survey paper needs a number, even if it's "5–50× slower than `FusedMoE`".

### 2.8 Granite (section 8)

- The four multipliers list (target md lines 858–863) is correct against the Granite-3.1-3B config but the *value* of `attention_multiplier` is unusual: `0.0078125 ≈ 1/128`. The report writes `≈ 1/(8*16)` which is technically `1/128` but the *muP* intuition is `1/(2*head_dim)` for some Granite variants. Without a clearer derivation, readers can't follow.
- llama.cpp pseudocode (target md lines 874–890): the `cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale)` on the final logits line is *correct*, but the comment on target md line 892 says the embedding scale "is done inside `build_inp_embd`" — which is true in master but was *not* always true historically; some older granite GGUFs bake the scale into the weights. Worth a footnote.

### 2.9 OLMo 2 (section 9)

- The Q/K-norm dimension claim (target md lines 936–943) is verified at HF source lines 231–232. The contrast with Qwen3's `Dh`-only gamma is correctly drawn.
- The "no input_layernorm" claim (target md line 911) is verified at HF source lines 295–304.
- But the *pseudocode formulation* on target md line 933 (`OLMo2: x_out = x + post_norm(sublayer(x))`) misses a subtle point: `sublayer(x)` for attention includes Q/K-norm applied to projections of `x`, which behaves like an embedded normalization. So OLMo2 *does* normalize Q and K (via q_norm/k_norm) before they hit attention; it just doesn't pre-norm the residual stream. The pseudocode flattens away this distinction and could mislead a reader into thinking OLMo2 attention is "fully unnormalized".

### 2.10 Jamba (section 10)

- The MambaMixer sketch (target md lines 1019–1051) gets the structure right but several *shape annotations are wrong or missing*:
  - `proj = self.in_proj(hidden_states).transpose(1,2)` — input is `[B, S, D]`, output is `[B, S, 2*E_int]`, transpose gives `[B, 2*E_int, S]`. The pseudocode says `[B, 2*E_int, S]` in the comment but the variable name `proj` doesn't carry the shape. A survey should always carry shapes.
  - `causal_conv1d_fn(x, self.conv1d.weight.squeeze(1), self.conv1d.bias, "silu")` — the `weight.squeeze(1)` is needed because `Conv1d` weight has shape `[E_int, 1, kernel_size]` for depthwise (groups=E_int). Worth annotating.
  - `dt = self.dt_proj(dt).transpose(1,2)` — input `[B, S, dt_rank]` → `Linear → [B, S, E_int]` → transpose → `[B, E_int, S]`. The pseudocode is correct but the shape annotation only appears in the inline comment, easy to miss.
  - `A = -torch.exp(self.A_log.float())` — but actually it's `A = -A_log.float().exp()` in the source; same math, different stylistic order. Minor.
- The "Mamba KV cache" naming on target md line 1062 — `conv_state` and `ssm_state` — is HF's naming; vLLM calls them `conv_state_tensor` and `ssm_state_tensor` with paged-style management. The report should note that the *cache abstraction* itself differs across engines.
- `JambaSparseMoeBlock` (line 533 of source per the cite) is referenced but no pseudocode is shown. A survey paper should at least sketch it (it's structurally the same as Mixtral's, but Jamba alternates expert layers as well as attention/Mamba layers).

---

## Section 3 — Divergence completeness gaps (HF / vLLM / llama.cpp)

The report's "Surprising HF / vLLM / llama.cpp divergences" list (15 items) is good as far as it goes, but several important divergences are missing or under-developed:

1. **llama.cpp's QK weight permutation for RoPE basis flip.** llama.cpp's GGUF conversion code permutes Q and K weights to align with its `rope_type=NEOX` vs `NORM` convention. The HF `modeling_*.py` files use a particular `rotate_half` definition; ggml's RoPE uses a permuted basis. The conversion script in `llama.cpp/convert_hf_to_gguf.py` does `permute(weight, n_head, n_head_kv)` to make the math equivalent. The report mentions `rope_type=GGML_ROPE_TYPE_NEOX` (target md line 98) but never mentions that the *weights themselves* are permuted at conversion time. This is THE classic footgun when porting models — it deserves a section.

2. **vLLM continuous batching implications on the layer.** The report's pseudocode shows vLLM ops as if they operate on `[B, S, D]` (e.g., target md line 119: `qkv, _ = self.qkv_proj(hidden_states)`). In actual vLLM, `hidden_states` is `[T, D]` where `T = sum of (active prompt lens + 1-for-decode for each running request)`. This changes:
   - The attention call's mask: `Attention(...)` takes per-request block tables, not a `[B, S]` mask.
   - The RoPE call: `positions` is `[T]`, not `[B, S]`, so cos/sin gather is per-token.
   - The cache: paged, not contiguous; `kv_cache.update` is replaced by writing to the per-request block.

   None of this is in the divergence list. A survey paper has to show that vLLM's "Llama is the same Llama" pseudocode is *not* operating on the same input shape as HF's. Otherwise the comparison is misleading.

3. **ggml's quantized matmul integration.** When the GGUF tensor is q4_0 or q4_K_M, the `ggml_mul_mat` op dispatches a *dequantize-and-multiply* kernel, not a plain GEMM. This affects:
   - The shape of stored weights: q4_0 stores 32 weights per "block" with one shared scale; the storage shape is `[D/32, Df, 32 bytes+overhead]`, not `[D, Df]`.
   - The lazy-loading and tensor-split conventions.
   - The fact that a "layer" in ggml is really a graph node, and quantization is a property of the *node's input*, not of the math.

   The report mentions quantization not at all in section 02. If this is reserved for section 04, the cross-reference should be explicit.

4. **vLLM tensor-parallel sharding axis.** `QKVParallelLinear` shards the output dim across TP ranks; `RowParallelLinear` shards the input dim and all-reduces. The report mentions these names but never explains the sharding axis or the `tp_size` divisibility constraints (e.g., `num_heads` must be divisible by `tp_size`, `num_kv_heads` may need to be repeated for small TP, etc.). A survey paper on layer implementations CANNOT skip the TP story.

5. **FlashAttention vs SDPA backend selection.** HF's `ALL_ATTENTION_FUNCTIONS.get_interface(...)` is mentioned once (e.g., target md line 273) but the *backend-selection logic* and what each backend does differently (eager: pure pytorch, sdpa: torch's fused, flash_attention_2: FA2 lib, flash_attention_3: FA3 lib, flex_attention: torch 2.5+) is never discussed. Each backend has different support for:
   - Sliding window (FA2.5+ only)
   - Softcap (FA2.5.7+ only — flagged at target md line 1200)
   - GQA repeat (SDPA repeats internally, eager via `repeat_kv`)
   - Sink tokens (FlexAttention or custom mask only)

6. **HF's mask construction vs vLLM's block-table.** HF builds `attention_mask` as a `[B, 1, S, S]` (or `[B, 1, S, S+past]`) fp32 tensor of 0 and -inf. vLLM builds a per-request block table with paged indices. llama.cpp builds a `kq_mask` tensor. The three masks are *not interchangeable*. The report glosses over this.

7. **Phi-3 fused QKV claim verification.** The report claims (target md line 487) that "Phi3ForCausalLM inherits from LlamaForCausalLM with packed_modules_mapping". This is plausibly true but no actual vLLM file line is cited. For a survey paper, spot-checking would tell us whether the same is true for the long-context Phi-3-128k variant (which has LongRoPE → different rotary backend).

---

## Section 4 — Evolution narrative gaps

The report has no "evolution" section, full stop. It has cross-sectional comparisons (norm placement table in Appendix B; bias table in Appendix D) but no longitudinal one. A 3-year-evolution survey needs the following missing arcs:

1. **Llama 1 → Llama 2 → Llama 3.** Specifically:
   - Llama 1: MHA, 32k vocab, 2k context, sentencepiece tokenizer, no GQA.
   - Llama 2: MHA in 7B/13B, GQA in 34B/70B (`num_kv_heads=8`), 4k context, 32k vocab unchanged.
   - Llama 3: GQA everywhere (8B has `Hk=8`, 70B has `Hk=8`), 128k vocab (4× growth!), 8k native context with RoPE-scaling to 128k.
   - WHY: GQA reduces KV-cache by `H/Hk` for the same quality (Ainslie et al. 2023); 128k vocab improves rare-token compression (Llama-3 tokenizer is essentially OpenAI's tiktoken).
   - The report never mentions Llama 1 or Llama 2 at all. There's a sentence on Llama 3 405B biases. That's it.

2. **GPT-J / GPT-NeoX parallel-residual experiments.** GPT-J (2021), GPT-NeoX (2022), Falcon-7B (2023) all used parallel attn+FFN: `x_out = x + attn(norm(x)) + mlp(norm(x))`. This was tried, found to be quality-neutral but engineering-awkward (no overlap of attn and mlp), and dropped in Llama. This is a *dead end* the report should mention to make the "why" of pre-norm-with-sequential-residual cleaner.

3. **Phi-1 → Phi-2 → Phi-3 → Phi-4.** Specifically:
   - Phi-1 / Phi-1.5: 1.3B / 1.5B, MHA, GELU MLP, parallel attn+FFN inherited from CodeGen.
   - Phi-2: 2.7B, switched to standard pre-norm sequential residual.
   - Phi-3: SwiGLU MLP, GQA, fused QKV, partial RoPE, residual dropout.
   - Phi-3.5-MoE: Top-2 routing, 16 experts.
   - Phi-4: 14B dense, then Phi-4-mini-flash with Samba (Mamba + sliding-window attention).
   - The report covers Phi-3 only. The arc is essential to the narrative.

4. **Qwen 1 → Qwen 1.5 → Qwen 2 → Qwen 2.5 → Qwen 3.** Each transition introduced architectural changes:
   - Qwen 1: MHA, ALiBi (some variants), Q/K/V biases.
   - Qwen 1.5: dropped ALiBi for RoPE.
   - Qwen 2: GQA, fused gate-up, dropped some biases.
   - Qwen 2.5: minor refinements.
   - Qwen 3: QK-norm, per-layer SWA option, Qwen3-Next adds gated DeltaNet.
   - Report covers Qwen 3 only.

5. **Mistral 7B → Mixtral 8x7B → Mistral-Large → Mistral-Nemo / Codestral → Mistral-Small-3.** Mistral's architectural choices have shifted significantly across versions (Mistral-Large dropped SWA; Codestral adds FIM). Report covers one snapshot.

6. **MoE genealogy: Switch → GShard → Mixtral → DeepSeek-V2 → DeepSeek-V3 → Qwen3-MoE.** Each step shifted the routing function (Switch: top-1; GShard: top-2 with auxiliary loss; Mixtral: softmax+renorm; DeepSeek-V2: softmax+aux loss; DeepSeek-V3: sigmoid+bias correction, no aux loss; Qwen3-MoE: softmax+renorm with dense-MoE alternation). The "why" of each change is research-publishable. Report covers only two endpoints (DeepSeek-V3 and Mixtral) without the lineage.

7. **MLA introduction (DeepSeek-V2) and convergence on it.** MLA was introduced in DeepSeek-V2 (May 2024), refined in DeepSeek-V3. Other groups have *not* adopted MLA (cost of weight conversion). Why? Because the conversion-and-absorb logic is engineering-heavy. The report mentions absorb-into-`o_proj` but not the *cost* of doing this analysis or *why MLA hasn't spread*.

8. **Sliding-window attention adoption arc.** Longformer (2020) → Mistral 7B (Sep 2023, all-layers SWA) → Mixtral (config has SWA but model ignores it) → Gemma 2 (5:1 SWA pattern) → Gemma 3 (kept the 5:1 + softcap + double norm) → Qwen 3 (configurable per-layer). Why 5:1 specifically? Why did Mixtral drop it? Where's the citation? Survey paper bar: this needs a paragraph.

9. **Embedding tying / decoupling.** Llama-1 tied embeddings. Llama-2/3 decoupled them. Gemma-2 ties. Gemma-3 ties. Cohere ties+scales. Granite decouples and scales. SmolLM3 decouples. This is a *per-family* choice that affects parameter count materially (`vocab_size * D` extra parameters). Not in the report.

10. **Dead ends section is missing.** Survey papers traditionally have a "dead ends" section that lists tried-and-abandoned ideas:
    - Parallel attn+FFN (mentioned above).
    - ALiBi (Baichuan-1 → Baichuan-2 dropped it).
    - Yi-1.0's cosine-similarity attention.
    - Mistral-7B's `sliding_window=4096` that Mixtral kept-but-ignored.
    - DeepSeek-V2's `auxiliary_loss` for MoE (replaced by V3's no-aux bias correction).

---

## Section 5 — Correctness errors found (with citations)

I picked three family sections and verified them against the local HF transformers clone at `C:\Users\zhengte\external\transformers\src\transformers\models\`.

### 5.1 Qwen3 QK-norm placement (target md section 2a)

**Claim under test:** "The QK-norm is per head_dim, applied after the projection's view-into-heads and before RoPE." (target md line 228)

**Source check (`modeling_qwen3.py` lines 263–268, verified):**
```python
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

**Verdict: CORRECT.** `q_norm` operates on shape `[B, S, H, Dh]` (after `view`, before `transpose`), with `gamma.shape = [Dh]`. RoPE is applied after the transpose. The report's claim is accurate.

**Minor:** The report should note that the Qwen3 source comment at line 248 also explicitly contrasts with OLMo ("unlike olmo, only on the head dim!"). The report quotes this on line 943 (under OLMo 2 section) but doesn't quote it under Qwen 3 itself, which is the source of the comment.

### 5.2 Gemma 3 sandwich norm placement (target md section 3a)

**Claim under test:** "Four RMSNorms per layer (vs Llama's two). The 'post' norms run inside the residual branch — `residual + norm(sublayer(norm(x)))`. This is the 'sandwich norm' pattern from Gemma 2." (target md line 330)

**Source check (`modeling_gemma3.py` lines 407–426, verified):**
```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(...)
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = self.pre_feedforward_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

**Verdict: STRUCTURALLY CORRECT** — the report's formula `x_out = x + post_norm(sublayer(pre_norm(x)))` matches the source. The label "sandwich norm" is community-conventional but **terminologically imprecise**: the original "sandwich norm" from CogView (Ding et al. 2021) wraps the residual on BOTH sides (`norm(x + sublayer(norm(x)))`). Gemma 3's pattern is more accurately called "double pre-post norm" or "DeepNet-style" norm. A survey paper should use precise terminology.

**Correctness error in summary table (Appendix B, target md line 1401):** The table column "Post-attn norm" for Gemma 3 says `post_attention_layernorm`. This is correct, but the column "Pre-attn norm" says `input_layernorm` and the column "Pre-ffn norm" says `pre_feedforward_layernorm`. The reader has to know that the post_attention_layernorm runs *inside* the residual branch (not after the residual add) to interpret this row correctly — but the table doesn't show that. The table is structurally misleading without a legend.

### 5.3 DeepSeek-V3 MLA cache compression (target md section 6a)

**Claim under test:** "The cached representation is **already decompressed** here (`key_states` and `value_states` after cat). That's HF's simple-correct path." (target md line 638)

**Source check (`modeling_deepseek_v3.py` lines 416–479, verified):**
- Line 432: `q_states = q_states.view(query_shape).transpose(1, 2)` — q_states shape `[B, H, S, qk_head_dim]` where `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`.
- Line 438: `k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)` — `key_shape = (B, S, -1, qk_nope_head_dim + v_head_dim) = (B, S, H, 256)`.
- Line 441: `k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)` — single-head k_rot.
- Line 451: `key_states = torch.cat((k_pass, k_rot), dim=-1)` — shape `[B, H, S, qk_head_dim] = [B, H, S, 192]`.
- Line 454: `key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)`.

**Verdict: CORRECT.** HF caches the decompressed key (`[B, H, S, qk_head_dim]`) and value (`[B, H, S, v_head_dim]`). Per-token cache size is `H * (qk_head_dim + v_head_dim) = 128 * (192 + 128) = 40960 elements`. The "compressed" form would be `kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576 elements per token`. Ratio is `40960 / 576 ≈ 71×` — far larger than the "10×" the report mentions on line 638. **The report's 10× number is wrong.** The correct factor for DeepSeek-V3's 128-head config is ~71×; for V2's smaller-head configs it's smaller but still >10×. This is a real numeric error that should be fixed.

### 5.4 (Bonus) Other corrections found while skimming

- Target md line 1442 (Appendix D row for OLMo 2): Q/K/V/O biases all marked ✗. Verified at `modeling_olmo2.py` lines 220–229: `bias=config.attention_bias`. The default config has `attention_bias=False`, so ✗ is the typical case, but the table should say "✗ (config-dependent)" or note that the *flag exists*.
- Target md line 1432 (Appendix D row for Llama 3 8B): Q/K/V/O all ✗. Verified at `modeling_llama.py` source: bias is `config.attention_bias`, default False for 8B. Same caveat as OLMo2.
- Target md line 1457 (Appendix E row for Gemma-3-4B): "rms_norm_eps = 1e-6". The Gemma 3 config actually uses `rms_norm_eps = 1e-6` for the *text* norm and a separate `rms_norm_eps` for the vision norm. The table doesn't distinguish.
- Target md line 1444: "DeepSeek's `e_score_correction_bias` (non-trainable buffer)" listed in the "Router bias" column for DeepSeek-V3. This is correct in spirit, but `e_score_correction_bias` is technically a `nn.Buffer` (not a `nn.Parameter`), and the "non-trainable" claim deserves a citation to the DeepSeek-V3 paper (Section 4.2 "Auxiliary-Loss-Free Load Balancing").

---

## Section 6 — Recommended fixes for v2

In priority order:

1. **Add at least 8 more CRIT-severity families.** The minimum survey-paper bar is:
   - BitNet b1.58 (ternary weights, dual sub-norms)
   - OpenELM (per-layer width)
   - Mamba-2 (distinct from Jamba's Mamba-1)
   - RecurrentGemma / Griffin (RG-LRU)
   - Falcon-Mamba / Falcon-H1
   - Zamba 2 (parameter-shared attention)
   - Qwen3-Next (gated DeltaNet)
   - RWKV-7 (non-transformer recurrent)
   - Falcon-7B (parallel residual — the dead-end exemplar)
   - Phi-4-mini-flash / Samba (Mamba + SWA + MLP)

2. **Add the evolution-narrative section.** For each lineage (Llama, Phi, Qwen, Mistral, MoE, MLA, SWA), write a half-page that:
   - Lists the architecture diff per generation
   - Cites the paper / blog announcement
   - Explains *why* the change was made (quality, compute, KV-cache, etc.)
   - Lists dead ends

3. **Fix the 10× → 71× cache compression number for DeepSeek-V3.** Provide a worked example with the actual numbers `H=128, qk_head_dim=192, v_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64`.

4. **Disambiguate sandwich norm terminology.** Either use "double pre-post norm" or define "sandwich norm" precisely. Cite the CogView paper.

5. **Expand the divergence list with the 7 missing items in Section 3** of this critique: QK weight permutation, continuous batching shape change, ggml quantized matmul, vLLM TP sharding axis, attention backend selection, mask format trinity, Phi-3-128k variant.

6. **Add a "new axes" section** for axes not in the 20-axis taxonomy:
   - Per-layer width/heads (OpenELM)
   - Sub-norms inside attention output (BitNet `attn_sub_norm`) and inside MLP (`ffn_sub_norm`)
   - Parameter sharing (Zamba2)
   - Parallel residual (Falcon-7B / GPT-J)
   - Embedding-LM-head tying choice
   - Logit softcap (Gemma 2 final layer, not just attention softcap)
   - No-position-encoding (NoPE) layers (SmolLM3 alternation)
   - Sink tokens (GPT-OSS)

7. **Make tensor shapes mandatory in every pseudocode block.** Every variable assignment should carry a `# [B, S, ...]` annotation. Many of the issues in Section 2 above stem from missing or partial shape annotations.

8. **Cross-check every "lines XX–YY" citation against the actual files.** Especially for vLLM and llama.cpp where the report admits ("Things I couldn't access cleanly") that the cites are not always direct reads. A survey paper cannot have unverifiable citations.

9. **Add a table of "what each engine optimizes that the others don't"** distinct from the math-divergence table — this is the engineering story the report half-tells:
   - HF: math correctness, debuggability, dynamic shapes.
   - vLLM: fused kernels, paged KV cache, continuous batching, TP/PP/EP sharding.
   - llama.cpp: quantization, CPU/GPU portability, GGUF unified file format, low memory.

10. **Move Appendix A's universal pseudocode to the front (Section 0)** and frame the rest of the doc as "axes that perturb this base block". Currently the universal pseudocode reads as a *conclusion*, but it should be the *coordinate system* a reader uses while reading the rest.

11. **Add quantitative tables.** A survey paper has numbers:
    - KV cache size per token (in bytes) for each family.
    - Parameter count for each layer type.
    - FLOPs per token per layer.

12. **Fix the citation to "llama.cpp #29402"** which is actually `transformers#29402` (target md line 403).

---

## Final summary

The current document is a strong technical note covering ~10 Llama-family decoder variants from the HF / vLLM / llama.cpp angle. To meet the bar of a "survey paper analyzing LLM-layer evolution over 3 years", it needs:

- **At least 2.5–3× more families** (BitNet, OpenELM, Mamba-2, RecurrentGemma, Falcon-Mamba, Zamba2, Qwen3-Next, RWKV-7, Falcon-7B, Samba/Phi-4-mini-flash, etc.)
- **A longitudinal evolution narrative** with per-lineage architecture diffs and citations.
- **Numerical corrections** (DeepSeek-V3 MLA compression ratio: 10× → 71×, possibly more).
- **Stricter terminology** (sandwich norm, partial vs interleaved RoPE, post-norm vs pre-norm-plus-sublayer-output-norm).
- **Engineering-divergence completeness** (continuous batching, QK permutation, ggml quantized matmul, FlashAttention backend selection, TP sharding).
- **New architectural axes** (per-layer width, sub-norms, parallel residual, embedding tying, NoPE, sink tokens).

Without these, the document is a competent ten-family snapshot. With them, it would be a survey paper.
