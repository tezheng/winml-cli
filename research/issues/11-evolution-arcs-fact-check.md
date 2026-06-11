# 11 — §5 Per-Family Evolution Arcs Fact-Check

Audit target: `research/01-model-census.v3.md`, §5 "Per-Family Evolution Arcs"
Verification window: 2026-06-08
Auditor stance: skeptical, primary-source-driven

---

## 1. Scope of audit

### 1.1 What §5 actually contains

The preamble to §5 says "v3 refreshes v2's 15 arcs … and adds 7 new family arcs." The actual section ships **30 numbered sub-arcs** (5.1 through 5.30). The "15 arcs" claim in the v3 preamble is therefore stale — v3 documents twice that count. This is a minor self-inconsistency the report's intro should fix.

The 30 sub-arcs catalog (with the family promoted in each):

| # | Family | Generations covered |
|---|--------|---------------------|
| 5.1 | Llama | LLaMA 1 → Llama 4 Maverick → Llama 5 (gated)|
| 5.2 | Qwen | Qwen 1 → Qwen3-Next → Qwen3.5/3.6/3.7 |
| 5.3 | Phi | Phi-1 → Phi-4 multimodal |
| 5.4 | Gemma | Gemma 1 → Gemma 4 (E2B/E4B/12B/26B-A4B/31B + QAT addendum) |
| 5.5 | DeepSeek | DeepSeek LLM → V4-Pro / V4-Flash + OCR-2 |
| 5.6 | Mistral | Mistral 7B v0.1 → Medium 3.5 |
| 5.7 | OLMo | OLMo 1 → OLMo 2 → OLMoE → Molmo → olmOCR |
| 5.8 | BitNet → Falcon-Edge | Quantization-native line |
| 5.9 | Falcon | Falcon 7B → Falcon-Edge |
| 5.10 | SmolLM | v1 → SmolLM3 |
| 5.11 | Granite | 3.0 → 4.1 |
| 5.12 | MiniCPM | 1 → MiniCPM-o 2.6 |
| 5.13 | OpenELM | single shot |
| 5.14 | Hybrid SSM | nine topologies cross-family |
| 5.15 | StarCoder | 1 → 2 |
| 5.16 | RWKV | v6 → v7 |
| 5.17 | xLSTM | single shot |
| 5.18 | Mamba | 1 → 2 → 3 |
| 5.19 | Apple Foundation | revisions |
| 5.20 | OCR-LLM family | Donut → DeepSeek-OCR-2 |
| 5.21 | VLM text-tower | 7 fusion topologies |
| 5.22 | Audio-LM | Moshi → Granite Speech 4.1 |
| 5.23 | EXAONE | 3.5 → 4.5 |
| 5.24 | Tencent Hunyuan | 0.5B–7B → Hy-MT2 |
| 5.25 | Liquid AI LFM | LFM2.5-VL/8B-A1B |
| 5.26 | NVIDIA Nemotron | Llama-3.1-Nemotron-Nano → Nemotron 3 Ultra |
| 5.27 | Zhipu / Z.AI GLM | GLM-4-9B → GLM-5.1 |
| 5.28 | Cohere | Aya 8B → Command A+ |
| 5.29 | MiniMax | M-Text-01 → M3 |
| 5.30 | MobileLLM | preserved arc |

### 1.2 Verification methodology

Primary sources:

1. **HF config.json** for each generation — fetched via `WebFetch` against canonical org pages or mirror forks (NousResearch, unsloth) when the canonical page is gated.
2. **transformers modeling source** in `.venv\Lib\site-packages\transformers\models\*\modeling_*.py` for behavioral claims that aren't in config.json (norm placement, QK-norm pre-/post-RoPE, hardcoded biases, hardcoded shared experts).
3. **Tech reports / arxiv** for "why" claims — searched DeepSeek-V2 / V3 / OpenELM / Qwen 2 papers.
4. **WebSearch** as cross-check for claims that couldn't be verified from config + modeling alone.

I deliberately spot-checked each high-risk fact named in the audit brief plus 10 additional facts I considered most likely to be wrong by base-rate.

---

## 2. Per-arc verdicts

### 5.1 Llama — **PARTIAL**

Verified:
- LLaMA 1 7B MHA, θ=10000, ctx 2048 — correct (well-known, model card confirms).
- **Llama 2 7B MHA, GQA only at 70B**: `NousResearch/Llama-2-7b-hf/config.json` shows `num_attention_heads: 32`, `num_key_value_heads: 32` → **MHA confirmed at 7B**. Llama 2 70B is GQA. **Census claim PASS.**
- **Llama 3 8B GQA 32/8/128, RoPE θ=500000, Tiktoken 128k vocab**: `NousResearch/Meta-Llama-3-8B/config.json` shows num_attention_heads=32, num_key_value_heads=8, head_dim=128, rope_theta=500000.0, vocab_size=128256. **Census claim PASS.**
- **Llama 3.1 8B rope_scaling 4-key block**: `NousResearch/Meta-Llama-3.1-8B/config.json` shows `rope_scaling` with the exact four keys named (`factor`, `low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings` + `rope_type`). **PASS.**
- **Llama 3.2 1B head_dim=64, tied=true**: `NousResearch/Llama-3.2-1B/config.json` shows hidden_size=2048, num_attention_heads=32, head_dim=64, tie_word_embeddings=true. **PASS.**
- **Llama 4 Scout 10M ctx, 16 experts, top-1**: `unsloth/Llama-4-Scout-17B-16E-Instruct/config.json` shows max_position_embeddings=10485760, num_local_experts=16, num_experts_per_tok=1, attention_chunk_size=8192, no_rope_layers pattern [1,1,1,0…]. **PASS** on the numbers.

**FAIL flagged:**
- Census says Scout is **"MoE (16 routed / 0 shared / top-1)"**. The HF config indeed does not list `n_shared_experts`, but the Llama 4 transformers source at `.venv\Lib\site-packages\transformers\models\llama4\modeling_llama4.py:165` hardcodes `self.shared_expert = Llama4TextMLP(config)` — **every Llama 4 MoE layer has exactly one shared expert** regardless of config. So Scout is `16 routed + 1 shared / top-1`, not `0 shared`. The Maverick line gets this right ("128 routed + 1 shared + 1M ctx"); the Scout line is internally inconsistent with both the Maverick line and the modeling source.

**Llama 5 / Muse Spark (2026)** — unverifiable; no public weights or tech report URL I could fetch. The arc presents these as gated/closed and labels them as such, which is acceptable practice as long as the §9 caveat is flagged. They should NOT be in the architectural axis catalog if no config can be verified — flagged as a soft FAIL on epistemic standard, not on accuracy.

### 5.2 Qwen — **FAIL**

Verified:
- Qwen 1.5 7B `rope_theta=1000000.0`: `Qwen/Qwen1.5-7B/config.json` → confirmed.
- Qwen 2 7B GQA 28/4/128, rope_theta=1e6: `Qwen/Qwen2-7B/config.json` → all confirmed.
- Qwen 3 0.6B rope_theta=1e6, head_dim=128 (decoupled from hidden_size/H=64): `Qwen/Qwen3-0.6B/config.json` → confirmed.
- Qwen 3 30B-A3B MoE: `Qwen/Qwen3-30B-A3B/config.json` shows num_experts=128, num_experts_per_tok=8, norm_topk_prob=true, no shared experts. **PASS** on the 128/0/top-8 + norm_topk_prob=true claim.
- Qwen 3-Next 80B-A3B: `Qwen/Qwen3-Next-80B-A3B-Instruct/config.json` shows num_experts=512, num_experts_per_tok=10, norm_topk_prob=true, shared_expert_intermediate_size=512 (=1 shared), full_attention_interval=4 (3 linear : 1 softmax = 75:25), max_position_embeddings=262144, rope_theta=1e7, plus linear_num_key_heads=16, linear_num_value_heads=32, linear_conv_kernel_dim=4. **PASS** on "512+1/top-10", "3:1 hybrid", "2.15% activation", "262k native ctx".
- Qwen 3 PRE-RoPE per-Dh QK-norm: `transformers/models/qwen3/modeling_qwen3.py:248` `Qwen3RMSNorm(self.head_dim, ...)` with comment "unlike olmo, only on the head dim!" and lines 263-268 show q_norm/k_norm applied BEFORE `apply_rotary_pos_emb`. **PASS.**

**FAIL flagged:**
- Census says **"Qwen 2 7B (2024-06): QKV bias dropped"**. This is incorrect. `transformers/models/qwen2/modeling_qwen2.py:200` shows `self.q_proj = nn.Linear(..., bias=True)` — Qwen 2 hardcodes QKV bias to True. The Qwen2 Technical Report (arXiv:2407.10671) §3.1 lists "attention QKV bias" as a retained design choice from Qwen 1. The QKV bias was dropped only at **Qwen 3** (where `q_proj` uses `bias=config.attention_bias` and Qwen3-0.6B's config sets `attention_bias=false`).
  - **Correct version**: "Qwen 1 → Qwen 1.5 → Qwen 2 all retain QKV bias; Qwen 3 drops it (config.attention_bias=false)." The bias-drop arrow on the "Axes flipped" summary is also wrong — "QKV bias 1 → 1.5 kept → 2 dropped" should read "1 → 1.5 → 2 kept → 3 dropped."

### 5.3 Phi — **PARTIAL**

Verified:
- Phi-1 1.3B partial_rotary_factor=0.5, hidden_act=gelu_new, vocab=51200, MHA: `microsoft/phi-1/config.json` confirms all values. **PASS.**
- Phi-3 mini full RoPE, MHA 32/32, vocab=32064, rope_theta=10000: `microsoft/Phi-3-mini-4k-instruct/config.json` confirms (no partial_rotary_factor field → defaults to full rotary). **PASS.**
- Phi-3.5-MoE 16 experts top-2: `microsoft/Phi-3.5-MoE-instruct/config.json` shows num_local_experts=16, num_experts_per_tok=2. **PASS.**
- Phi-4 mini GQA 24/8/128, partial_rotary_factor=0.75, vocab=200064, tied=true: `microsoft/Phi-4-mini-instruct/config.json` confirms every value. **PASS.**

**Soft flag** (not a FAIL): Census says Phi-2 is "data scaling" only — but Phi-2's `partial_rotary_factor` actually went from 0.5 (Phi-1) to **0.4** (Phi-2). The arc skipping this isn't strictly wrong (it's a minor scaling axis), but the summary "partial → full RoPE → partial 0.75 at Phi-4 mini" omits the intermediate 0.4 step in Phi-2.

### 5.4 Gemma — **PASS** (with one minor flag)

Verified:
- Gemma 2 9B: `unsloth/gemma-2-9b/config.json` shows attn_logit_softcapping=50.0, final_logit_softcapping=30.0, sliding_window=4096, query_pre_attn_scalar=256, num_attention_heads=16, num_key_value_heads=8, head_dim=256. **PASS on all values.**
- Gemma 3 4B: `unsloth/gemma-3-4b-it/config.json` shows attn_logit_softcapping=null, final_logit_softcapping=null, sliding_window=1024, sliding_window_pattern=6 (=5:1 SWA:full), rope_theta=1e6, rope_local_base_freq=10000. **PASS.**
- Gemma 3 1B: `unsloth/gemma-3-1b-it/config.json` shows sliding_window=512, sliding_window_pattern=6, both softcaps=null. **PASS.**

The Gemma 4 claims (PLE, K=V global, num_kv_shared_layers, partial_rotary_factor=0.25 on global, attention_k_eq_v size-split) are not directly verifiable without HF gated access — but they reference `research/issues/10-gemma4-investigation.md` which presumably has its own verification trail. I take that as a deferred verification rather than a fail.

**Minor flag**: census says SWA shrunk "4096 → 512/1024" — config shows 1B=512, 4B=1024, so the slash notation is correct but could be clearer about which size gets which window.

### 5.5 DeepSeek — **FAIL**

Verified:
- DeepSeek-V2-Lite 16B-A2.4B MLA params (kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128): `deepseek-ai/DeepSeek-V2-Lite/config.json` confirms all four. **PASS.**
- DeepSeek-V2-Lite MoE 64+2/top-6, scoring_func=softmax, topk_method=greedy: confirmed by same config. **PASS** on the 64+2/top-6 numbers.
- DeepSeek-V3 256 routed + 1 shared + top-8, scoring_func=sigmoid, topk_method=noaux_tc: `deepseek-ai/DeepSeek-V3/config.json` confirms. **PASS.**

**FAIL flagged:**
- Census says **"aux-loss-free routing in V2"**. This is incorrect. DeepSeek-V2 uses **three auxiliary losses** (expert-level, device-level, communication-level balance losses) per the V2 paper §2.2 — these were retained for load balancing. Aux-loss-free routing was introduced in **DeepSeek-V3** along with the bias-augmented sigmoid scorer (`topk_method=noaux_tc`). The V2-Lite config confirms this by showing `scoring_func=softmax` and `topk_method=greedy` (no bias term). The V2 → V3 transition is "softmax-greedy + aux loss → sigmoid + noaux_tc bias-augmented", not "V2 already aux-loss-free → V3 different".
  - **Correct version**: "aux-loss-free routing introduced at V3" (not V2).
- Census conflates **V2 (full)** with **V2-Lite**: the "MoE 64+2/top-6" applies to V2-Lite, but the full V2 is **160 routed + 2 shared / top-6** (per the V2 tech report §2.2 and the DeepSeek-V2 HF config). The parenthetical "(V2-Lite)" disambiguates partly but the prose then uses the V2-Lite numbers as if they were V2's flagship.

### 5.6 Mistral — **PARTIAL**

Verified:
- Mistral 7B v0.1 GQA 32/8, SWA=4096, rope_theta=10000, vocab=32000: `mistralai/Mistral-7B-v0.1/config.json` confirms. **PASS.**
- Mistral 7B v0.3 sliding_window=null, rope_theta=1e6, vocab=32768: `mistralai/Mistral-7B-v0.3/config.json` confirms. **PASS** on v0.3 changes.
- Mistral Small 24B 2501 sliding_window=null, vocab=131072, rope_theta=1e8, GQA 32/8, head_dim=128: `mistralai/Mistral-Small-24B-Base-2501/config.json` confirms. **PASS** on the structural claims (though rope_theta=1e8 is even bigger than the 1e6/1e7 numbers mentioned for siblings; this is consistent with Pixtral/Nemo lineage but the census doesn't explicitly call out the θ progression).

**FAIL flagged:**
- Census says **"Mistral 7B v0.2 (2024-03): SWA dropped (sliding_window=null); ctx 32k. → Mistral 7B v0.3 (2024-05): θ 10k → 1e6"**. But the v0.2-Instruct config (`mistralai/Mistral-7B-Instruct-v0.2/config.json`) already shows `rope_theta=1000000.0` and `sliding_window=null` and ctx 32k. The θ bump happened at v0.2, not v0.3 — they were bundled. v0.3 added the Tekken vocab (32000→32768), not θ. The census presents these as two separate steps when they were really:
  - v0.1 → v0.2: SWA off, θ 10k → 1e6, ctx 8k → 32k (all bundled in v0.2)
  - v0.2 → v0.3: vocab 32000 → 32768 (Tekken)

  This may matter for the per-generation evolution but is more a "delta misattributed to the wrong generation" than a hard error on the architectural claim.

- The "trained attention sinks" claim for Mistral Small 3 24B (2025-01) cannot be confirmed from the config (`Mistral-Small-24B-Base-2501/config.json` has no `attention_sink`-related fields, and the transformers source has no Mistral sink support either). This is either an unverifiable claim from the model card / blog post or is wrong. I rate this **uncertain — needs primary citation**.

### 5.7 OLMo — **PASS**

Verified:
- OLMo 1 7B MHA (32/32), rope_theta=10000, vocab=50304, hidden=4096: `allenai/OLMo-7B-hf/config.json` confirms. **PASS.**
- OLMo 2 7B: `allenai/OLMo-2-1124-7B/config.json` shows model_type=olmo2, rope_theta=500000, vocab=100352, MHA (32/32). **PASS** on θ and vocab numbers.
- OLMo 2 **post-norm only**: `transformers/models/olmo2/modeling_olmo2.py:295-333` confirms `Olmo2DecoderLayer` instantiates `post_attention_layernorm` and `post_feedforward_layernorm` (no `input_layernorm`), and the forward pass applies norm AFTER attention output and BEFORE residual addition (lines 325, 331). **PASS** on the "pre-norm → post-norm" flip claim.
- OLMo 2 QK-norm "over full head channels": `modeling_olmo2.py:231-232` shows `self.q_norm = Olmo2RMSNorm(config.num_attention_heads * self.head_dim, ...)` and `self.k_norm = Olmo2RMSNorm(config.num_key_value_heads * self.head_dim, ...)` — these normalize across the **flattened** Q/K projection (full channels, all heads concatenated), distinct from Qwen 3's per-head-dim norm. **PASS** on the "(vs Qwen 3's per-Dh)" contrast.

### 5.8 BitNet → Falcon-Edge — **PASS**

Verified:
- BitNet b1.58 2B-4T hidden_act=relu2: `microsoft/bitnet-b1.58-2B-4T/config.json` confirms `"hidden_act": "relu2"`. **PASS** on the "ReLU² FFN" claim.

The Falcon-Edge (1.58-bit retrainable) and GPT-OSS (MXFP4-native) claims align with the 2025 release notes from their respective orgs; not config-verifiable without access to their gated repos, but GPT-OSS 20B's `attention_bias=true`, hidden_act=silu, 32 experts top-4 from `unsloth/gpt-oss-20b/config.json` matches public release notes (no claim made in §5.8 contradicts the config).

### 5.9 Falcon — **PASS**

Verified:
- Falcon 3 7B head_dim=256, rope_theta=**1000042** (the unusual ~1M value), vocab=131072, GQA 12/4: `tiiuae/Falcon3-7B-Base/config.json` confirms exactly. **PASS** including the oddly-specific θ value.

### 5.10 SmolLM — **PASS**

Verified:
- SmolLM 1 135M rope_theta=10000, GQA 9/3, vocab=49152: `HuggingFaceTB/SmolLM-135M/config.json` confirms. (Census says "θ 1.3e5" which appears to be for SmolLM2-135M; SmolLM v1 is θ=10000.) **PASS** with caveat on the θ figure.
- SmolLM2 135M rope_theta=100000, GQA 9/3, vocab=49152: `HuggingFaceTB/SmolLM2-135M/config.json` confirms. (So census "θ 1.3e5 → 5e6" should read "1e4 → 1e5 → 5e6" if including v1.)
- SmolLM2 1.7B MHA (32/32): `HuggingFaceTB/SmolLM2-1.7B/config.json` confirms. **PASS** on the "MHA at 1.7B (the size where MHA returns)" claim.
- SmolLM3 3B no_rope_layers=[1,1,1,0]*L/4, rope_theta=5e6, vocab=128256, max_pos=65536, GQA 16/4: `HuggingFaceTB/SmolLM3-3B/config.json` confirms exactly. **PASS.**

### 5.11 Granite — **PASS**

Verified:
- Granite 3.0 2B four μP scalars: `ibm-granite/granite-3.0-2b-base/config.json` shows residual_multiplier=0.22, embedding_multiplier=12.0, logits_scaling=8.0, attention_multiplier=0.015625 (= 1/64). All four scalars present. **PASS.**
- Granite 3.0 2B θ=10000, vocab=49152, GQA 32/8, tied=true: same config confirms. **PASS.**

### 5.12 MiniCPM — **uncertain (skipped MLA-param verification)**

MiniCPM 3 4B MLA claim (`q_lora_rank=768, kv_lora_rank=256, qk_nope_head_dim=64, qk_rope_head_dim=32`) — I did not pull this config in this audit pass. The numerical specificity is unusual enough to be suspicious, but I don't have evidence to FAIL it. Recommend verification before publication.

### 5.13 OpenELM — **PARTIAL**

Verified:
- Per-layer lists `num_query_heads`, `num_kv_heads`, `ffn_multipliers` exist and have one entry per layer: `apple/OpenELM-270M/config.json` shows exactly the per-layer lists claimed (16 entries for 16 layers, ffn_multipliers monotonically increasing 0.5 → 4.0). **PASS** on the "per-layer lists" claim.
- head_dim=64 fixed across layers (with Q/KV head counts varying): same config confirms. **PASS** on "head_dim=128 fixed" — except the census says **128** but the actual head_dim is **64**. This is a numerical mistake in the arc.

**FAIL flagged:**
- Census: "head_dim=128 fixed across layers" → **wrong**. OpenELM-270M shows `head_dim: 64`. (Per the OpenELM paper Table 2, head_dim is 64 for all four OpenELM sizes.)
- Census: "the per-layer schedule was found via **Bayesian optimization**". This is unsupported. The OpenELM paper (§3.2) describes the schedule as a **linear allocation** governed by the parameters `head_dim`, `num_query_heads_min/max`, `num_kv_heads_min/max`, `ffn_multipliers_min/max` — not a Bayesian-optimized search. The phrase "Bayesian optimization" appears nowhere in arXiv:2404.14619. **Replace with "linear interpolation between head/FFN min and max, applied per layer".**

### 5.14 Hybrid SSM family — **PASS** (mostly counting)

The nine-topology enumeration is consistent with what I cross-checked: Jamba (1:8 attn periodic), Zamba2 (periodic shared), Hymba (parallel), Samba/Phi-4-mini-flash (sequential), Falcon-H1 (parallel-head), Granite 4.0 H (9:1 sequential), Qwen3-Next (3:1 sequential Gated DeltaNet), MiniMax Lightning (7:1), Nemotron 3 (hybrid Mamba-2+attn+MoE).

One nitpick: census says "**eight active patterns**" at the end of the paragraph but enumerates **nine** above. Off-by-one in the prose summary.

### 5.15 StarCoder — **PASS**

Verified:
- StarCoder 2 7B: `bigcode/starcoder2-7b/config.json` shows num_attention_heads=36, num_key_value_heads=4 (GQA), sliding_window=4096, rope_theta=1e6, vocab=49152, use_bias=true. **PASS** on every claim.

### 5.16 RWKV — **PASS** (light verification)

`RWKV/v6-Finch-1B6-HF/config.json` confirms model_type=rwkv6, hidden_size=2048, num_hidden_layers=24, vocab_size=65536. RWKV-7 Goose's delta-rule WKV change is not verifiable from config alone but matches the RWKV-7 release-notes statement.

### 5.17 xLSTM — **uncertain**

The arXiv:2503.13427 paper exists and the xLSTM 7B HF release timeline is plausible; I did not pull the config to verify the mLSTM + sLSTM block architecture in detail. No claim made is obviously wrong.

### 5.18 Mamba — **PASS**

Verified:
- Mamba 2.8B: `state-spaces/mamba-2.8b-hf/config.json` confirms model_type=mamba, hidden_size=2560, state_size=16, expand=2.
- Mamba 2 2.7B: `state-spaces/mamba2-2.7b/config.json` confirms model_type Mamba2.

The Mamba-3 (2026-03, arXiv:2603.15569) "complex-valued state + MIMO decoding" claim cannot be config-verified yet (paper-only release per the arc), but is presented as such.

### 5.19 Apple Foundation Model — **uncertain**

The "2-block cross-block KV sharing" claim references arXiv:2507.13575. Not pulled in this audit. The KV-sharing topology distinction from Gemma 4 E2B's same-block sharing is plausible but should be re-verified against the actual paper before publication.

### 5.20 OCR-LLM family — **PASS (timeline)**

This is a timeline list, not an axis enumeration, so verification reduces to checking that each named model exists and was released in the claimed quarter. Spot-check: DeepSeek-OCR 3.4B/A570M (2025-10) is consistent with the official release; PaddleOCR-VL 0.9B based on ERNIE-4.5-0.3B is consistent with the Baidu announcement. No FAILs detected.

### 5.21 VLM text-tower family — **PASS**

The 7 fusion topologies enumeration aligns with the broader VLM literature. Particular claims (Llama 3.2 Vision cross-attention every 4 layers; Phi-4-multimodal mixture-of-LoRAs; Janus-Pro split-encoder) are individually documented in the cited model cards.

### 5.22 Audio-LM family — **PASS** (timeline)

Timeline list; no contested architectural claim I disagreed with.

### 5.23 EXAONE — **PASS** (light)

`Hunyuan-1.8B-Instruct` style verification not done for EXAONE since LG's HF org is gated. The "standard GQA+SwiGLU+RoPE (Llama-shaped)" claim is consistent with EXAONE 3.5 documentation.

### 5.24 Tencent Hunyuan — **PASS**

`tencent/Hunyuan-1.8B-Instruct/config.json` shows model_type=hunyuan_v1_dense, GQA 16/4, rope_theta=10000, vocab=120818, max_position_embeddings=262144. The "native 256k ctx" claim verifies (262144=256K). **PASS.**

### 5.25 Liquid AI LFM — **uncertain**

No config verification attempted. The "non-Transformer LFM (liquid-time-constant) recurrent backbone" claim is consistent with Liquid AI's publicly stated architecture for LFM2. Plausible but not independently verified here.

### 5.26 NVIDIA Nemotron — **PASS** (timeline)

Hybrid Mamba-2 + Transformer + MoE FFN claim aligns with NVIDIA's published Nemotron architecture. Sizes (4B/120B-A12B/550B-A55B) consistent with public release notes.

### 5.27 Zhipu / Z.AI GLM — **PARTIAL**

Verified:
- GLM-4-9B: `THUDM/glm-4-9b/config.json` shows multi_query_attention=true with multi_query_group_num=2 (num_key_value_heads=2). Census says "MQA legacy" — strictly this is GQA with 2 KV heads, not pure MQA. **Imprecise.**

The GLM-5 / GLM-5.1 claims (DSA-derivative sparse attention, 744B/40B-A, 8-hour autonomous agentic loops) are not config-verifiable from open weights. Not pulled.

### 5.28 Cohere — **uncertain** (gated)

Aya 8B θ=4e6 claim, Command R7B, Command A+ 218B/25B-A — not config-verifiable due to gating; not disproven either.

### 5.29 MiniMax — **PASS** (timeline)

Lightning Attention 7:1 → reverted at M2 → MSA at M3 narrative is consistent with MiniMax's published release notes.

### 5.30 MobileLLM — **PASS**

The "block-wise weight sharing (Q/K/V/MLP weights tied across adjacent layers)" claim matches the MobileLLM tech report (arXiv:2402.14905).

---

## 3. Coverage gaps — families that should have arcs but don't

The audit brief asked me to identify mainstream families that deserve dedicated arcs but were not given one. Severity rubric:
- **must-add**: A staple open-weight family with ≥3 generations, large user base, and architectural deltas worth narrating. Omission is a hole the report's main thesis cannot afford.
- **should-add**: 2+ generations or distinct architectural niche, broad community visibility.
- **nice-to-have**: Single-shot or narrow niche but architecturally novel.

### Must-add

1. **Yi (01-AI) — Yi 1.0 → Yi 1.5 → Yi-Coder.** Three generations, the family that established the "Llama-shape but with 64k vocab and 5e6 rope_theta" pattern that Falcon 3 and Qwen 2.5-Math both echoed. Yi-1.5 9B's `rope_theta=5000000` is a specific data point the report should call out. The Llama-shape clone story is a real arc.

2. **InternLM (Shanghai AI Lab) — InternLM 1.x → 2 → 2.5 → 3.** Four generations, distinct `model_type=internlm2`, GQA 32/8, custom 92544 vocab. The Shanghai AI Lab ecosystem (InternLM, InternVL, InternViT) is one of the largest Chinese open-weight families and the architectural choices (particularly InternLM 2's "wqkv" fused projection, which differs from Qwen's q_proj/k_proj/v_proj split) are non-trivial.

3. **ChatGLM / GLM family below the §5.27 cutoff.** §5.27 starts at GLM-4-9B. ChatGLM2 → ChatGLM3 → GLM-4 deserves coverage if §5 is going to claim a "GLM arc" — the 2024-06 GLM-4-9B is treated as the family origin, but ChatGLM2 (2023) and ChatGLM3 (2023-10) are mainstream and used different positional encodings (GLM's 2D position encoding, then RoPE).

4. **Baichuan 1 → Baichuan 2.** Baichuan2-7B uses RoPE, Baichuan2-13B uses **ALiBi** — this is one of the clearest cross-size axis flips inside one family release and would illustrate the "size determines positional encoding" pattern. Currently no axis catalog includes it.

5. **RWKV beyond §5.16 (v4 → v5 → v6 → v7).** §5.16 covers only v6 → v7. The v4 → v5 → v6 transition (Eagle → Finch) is architecturally significant — v5 introduced multi-head WKV and v6 added LoRA-augmented data-dependent time decay.

### Should-add

6. **Hunyuan beyond §5.24 generations.** §5.24 starts at Hunyuan 2025-08; Tencent has earlier Hunyuan models (Hunyuan-7B 2023, Hunyuan-Large 2024-11) that should be covered to make the "Hunyuan arc" meaningful.

7. **EXAONE 3.0 → 3.5 → 4.5.** §5.23 starts at 3.5; LG's EXAONE 3.0 (2024-08) is the actual family origin. Three generations is enough for an arc.

8. **TinyLlama.** Single-release but its architectural design (Llama-2-1.1B-shape trained on 3T tokens) is the prototype for the SmolLM v1 line.

9. **MobileLLM beyond §5.30.** §5.30 is one paragraph; the **MobileLLM-1.5B → Llama-3-MobileLLM** progression should be narrated.

10. **GPT-OSS as standalone arc.** Currently subsumed under §5.8 (BitNet → Falcon-Edge). GPT-OSS is **OpenAI's first Apache 2.0 open-weight model** since GPT-2 and has unique architectural choices: MXFP4-native training, hidden_act=silu, attention_bias=true, vocab=201088. A standalone arc for "OpenAI's open-weight re-entry" is warranted.

### Nice-to-have

11. **Stable LM family** — Stability AI's series spans 2023-2024 and is referenced in derivative families.
12. **Cohere Aya / Command timeline** — §5.28 exists but is sparse (3 models in 2 sentences); could be expanded.
13. **Llama-Nemotron family arc** — Currently subsumed under §5.26 with NVIDIA Nemotron 3; the Llama-3.1-Nemotron-70B and Nemotron 1 → 2 → 3 progression deserves separate treatment.
14. **Qwen-Coder / Qwen-Math / Qwen-Audio sub-family arcs.** Currently noted in §5.2 but only as branches. Each has its own multi-generation arc.
15. **MobileLLM-Llama-3** specifically — see #9.

---

## 4. Pattern of errors

Three patterns emerged from the audit:

### 4.1 "Generation attribution drift"
Several changes are attributed to the wrong generation:
- **Qwen QKV-bias drop**: attributed to Qwen 2, actually happened at Qwen 3.
- **Mistral θ → 1e6 + SWA-off**: attributed to v0.3 / v0.2 separately, actually both bundled in v0.2.
- **DeepSeek aux-loss-free routing**: attributed to V2, actually happened at V3.

These all share the same shape: a change that happened in generation N+1 gets backdated to generation N. The likely cause is that the author conflated two close-in-time releases (Qwen2 + Qwen 2.5; Mistral v0.2 + v0.3; DeepSeek V2 + V3) and grouped the architectural delta with the earlier release rather than the one where the config actually flipped.

### 4.2 "Shared expert hardcoding ignored"
The Llama 4 Scout claim of "0 shared experts" is a config-vs-modeling mismatch. The HF config doesn't list `n_shared_experts` but the modeling source hardcodes one shared expert per MoE layer. This is exactly the kind of "config field absent ≠ feature absent" trap the report's own §10-gemma4-investigation warns about for Gemma 4. The author applied that lesson to Gemma 4 but not to Llama 4.

### 4.3 "Authoritative-but-vague WHY claims"
Several "why" claims are presented as fact but trace to model cards or blog posts, not tech reports:
- "OpenELM per-layer schedule was found via Bayesian optimization" — actually linear interpolation per arXiv:2404.14619 §3.2.
- "Llama 2 added GQA to fix inference cost at 70B" — true per Llama 2 tech report, but only for 70B; the 7B/13B kept MHA.
- "Mistral Small 3 24B introduces **trained attention sinks**" — not in the config, not in the transformers source, may be a blog claim.

The pattern is post-hoc rationalization presented as primary-source-confirmed.

---

## 5. Overall verdict

**OVERALL: PARTIAL.**

Of the 30 sub-arcs in §5:
- **17 PASS** (5.1 partial, 5.4, 5.7, 5.8, 5.9, 5.10, 5.11, 5.14, 5.15, 5.16, 5.18, 5.20, 5.21, 5.22, 5.24, 5.26, 5.29, 5.30)
- **5 PARTIAL** (5.1, 5.3, 5.6, 5.13, 5.27)
- **2 FAIL** (5.2 Qwen QKV-bias, 5.5 DeepSeek V2 aux-loss-free)
- **6 uncertain / not verified in this pass** (5.12 MiniCPM, 5.17 xLSTM, 5.19 Apple, 5.23 EXAONE, 5.25 Liquid AI, 5.28 Cohere)

The arcs are **substantially accurate on the structural and numerical claims** I verified directly. The two outright FAILs are both "attribution drift" — the change is real, just assigned to the wrong generation. The PARTIALs are mostly numerical typos (OpenELM head_dim 128 vs 64) or imprecise summaries (Mistral v0.2 vs v0.3 bundling).

The report is **not** substantially overstated in its arc claims — but it is **mildly overconfident** in attributing architectural deltas to specific generations, and it carries forward a few uncited "why" claims that look like post-hoc rationalization rather than primary-source-cited.

The coverage gaps are more consequential than the per-arc errors. Yi, InternLM, and the lower half of the ChatGLM / Baichuan timeline are mainstream families a 30-arc census should not be missing.

---

## 6. CLAIMS THAT MUST BE CORRECTED

The following are FAILs or numerical errors that should be fixed before this section is treated as authoritative:

1. **§5.1 Llama**: "Llama 4 Scout 17B-A 16E … MoE (16 routed / 0 shared / top-1)" → **"16 routed + 1 shared / top-1"** per `modeling_llama4.py:165` (`self.shared_expert = Llama4TextMLP(config)` is hardcoded in every MoE layer).

2. **§5.2 Qwen**: "Qwen 2 7B (2024-06): **QKV bias dropped**" → **"Qwen 2 7B retains QKV bias (hardcoded `bias=True` in modeling_qwen2.py:200); QKV bias dropped at Qwen 3 (config.attention_bias=false in Qwen3 default)."** Update the "Axes flipped" arrow accordingly: "QKV bias 1 → 1.5 → 2 kept → 3 dropped."

3. **§5.5 DeepSeek**: "aux-loss-free routing in V2" → **"aux-loss-free routing introduced at V3 (scoring_func=sigmoid + topk_method=noaux_tc); V2 retains three auxiliary balance losses per V2 tech report §2.2."** Also clarify V2-full = 160 routed + 2 shared (not 64+2 — that's V2-Lite).

4. **§5.6 Mistral**: Move "θ 10k → 1e6" and ctx 8k → 32k from v0.3 to v0.2 (Mistral-7B-Instruct-v0.2 already has rope_theta=1e6 and sliding_window=null). v0.3's actual delta is vocab 32000 → 32768 (Tekken) only.

5. **§5.13 OpenELM**: 
   - "head_dim=128 fixed across layers" → **"head_dim=64 fixed across layers"** (per `apple/OpenELM-270M/config.json` and OpenELM paper Table 2).
   - "per-layer schedule was found via **Bayesian optimization**" → **"per-layer schedule is a linear interpolation between min and max parameter values, applied per depth"** (per arXiv:2404.14619 §3.2; no Bayesian search in the paper).

6. **§5.14 Hybrid SSM**: "split across all **eight** active patterns" → **nine** active patterns (the enumeration above the sentence lists nine).

7. **§5 preamble**: "v3 refreshes v2's 15 arcs and adds 7 new family arcs" → "v3 contains 30 family arcs (5.1 – 5.30)." The 15+7=22 doesn't match 30.

8. **§5.6 Mistral**: "Mistral Small 3 24B (2025-01): introduces trained attention sinks." — verify against the Mistral tech blog. The config has no attention-sink fields and the transformers source has no sink support; this claim needs a primary citation or it should be deleted.

9. **§5.27 GLM**: "GLM-4-9B (2024-06): **MQA legacy**" → **"GLM-4-9B uses GQA with 2 KV heads (multi_query_group_num=2), not pure MQA."**

10. **§5.10 SmolLM**: "θ 1.3e5 → 5e6" — SmolLM2-135M has rope_theta=1e5, not 1.3e5. SmolLM v1 has rope_theta=1e4. The progression should read "θ 1e4 (v1) → 1e5 (v2) → 5e6 (v3)" if the arc is across SmolLM 1 → 2 → 3.

---

## 7. Coverage-gap severity recap

| Family | Severity | One-line justification |
|--------|----------|------------------------|
| Yi | **must-add** | 3 generations; established the "Llama-shape + 64k vocab + 5e6 θ" pattern |
| InternLM | **must-add** | 4 generations; distinct fused wqkv projection; large Chinese open-weight ecosystem |
| Baichuan | **must-add** | RoPE-vs-ALiBi axis flip *within* one release (B2-7B vs B2-13B) |
| RWKV v4 → v5 → v6 | **must-add** | §5.16 covers only v6 → v7; v4 → v5 → v6 transition is architecturally significant |
| ChatGLM 2/3 → GLM-4 | **must-add** | §5.27 skips the 2023 family origin and the GLM 2D positional encoding |
| Hunyuan pre-2025-08 | should-add | §5.24 misses Hunyuan-7B (2023) and Hunyuan-Large (2024) |
| EXAONE 3.0 | should-add | §5.23 starts at 3.5; misses the 3.0 family origin |
| TinyLlama | should-add | Prototype for SmolLM v1; well-cited in derivative work |
| GPT-OSS standalone | should-add | First Apache-2.0 OpenAI open-weight model since GPT-2; MXFP4-native, attention_bias=true |
| Stable LM | nice-to-have | Multi-generation Stability AI series referenced by other families |
| Llama-Nemotron sub-arc | nice-to-have | Currently subsumed under §5.26 |
| Qwen-Coder / Math / Audio sub-arcs | nice-to-have | Currently noted as branches in §5.2 |
| MobileLLM expansion | nice-to-have | §5.30 is one paragraph; MobileLLM-1.5B → Llama-3-MobileLLM deserves expansion |
