# Phase 1: Zero-Shot Baseline Evaluation - Work Log

> Last updated: 2026-03-22

## Overview

Phase 1 of the Medical VQA thesis: zero-shot baseline evaluation of 4 lightweight VLMs on 3 medical VQA datasets. This establishes baseline performance before QLoRA fine-tuning (Phase 2).

- **Hardware**: RTX 5060 Ti 16GB VRAM, Ryzen 5 5600X, 32GB RAM
- **Framework**: transformers 5.3.0, torch 2.10.0+cu128, omegaconf
- **Environment**: Windows 11, Python 3.12 (uv venv, downgraded from 3.13 for package compatibility)

---

## 1. Target Models

| # | Model | HF ID | Class | VRAM (FP16) | Status |
|---|-------|-------|-------|-------------|--------|
| 1 | Qwen3-VL-2B | `Qwen/Qwen3-VL-2B-Instruct` | `Qwen3VLForConditionalGeneration` | ~3.96 GB | **Working** |
| 2 | Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` | `Qwen2_5_VLForConditionalGeneration` | ~6.99 GB | **Working** |
| 3 | Florence-2-large | `microsoft/Florence-2-large` | `AutoModelForCausalLM` | ~2 GB | **Blocked** |
| 4 | SmolVLM2-2.2B | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `AutoModelForImageTextToText` | ~4.72 GB | **Working** |

---

## 2. Model Loading Test Results

### 2.1 SmolVLM2-2.2B - PASS

- **VRAM**: 4.19 GB (loading), 4.72 GB (peak during inference)
- **Prompt format**: `chat_template` (standard two-step: `apply_chat_template` -> `processor`)
- **dtype**: bfloat16
- **Test answer** (blank white image): "A white background."

**Bug fixed**: `_generate_standard_chat()` was passing `images=[image]` to both `processor.apply_chat_template()` and the subsequent `processor()` call, causing `"got multiple values for keyword argument 'images'"`. Fixed by separating into two steps:
1. `processor.apply_chat_template(messages, tokenize=False)` for text only
2. `processor(text=[text], images=[image], return_tensors="pt")` for tokenization

### 2.2 Qwen3-VL-2B - PASS

- **VRAM**: 3.96 GB
- **Prompt format**: `chat_template` with `qwen_vl_utils.process_vision_info()`
- **dtype**: float16
- **Test answer** (blank white image): "The image is completely blank and contains no visible content."
- **Dependency**: requires `qwen-vl-utils[decord]`

### 2.3 Qwen2.5-VL-3B - PASS

- **VRAM**: 6.99 GB
- **Prompt format**: `chat_template` with `qwen_vl_utils.process_vision_info()`
- **dtype**: float16
- **Test answer** (blank white image): "I cannot see any image as I am a text-based model..."

### 2.4 Florence-2-large - BLOCKED (transformers 5.x incompatibility)

- **VRAM**: ~2 GB (loads successfully)
- **Prompt format**: `direct_question` with `<VQA>` task prefix
- **dtype**: float16
- **Status**: Model loads, but generates garbage output

See [Section 4: Florence-2 Debugging](#4-florence-2-debugging-detailed) for full details.

---

## 3. VQA-RAD 10-Sample Evaluation

### 3.1 SmolVLM2-2.2B on VQA-RAD (10 samples) - PASS

```
closed_accuracy: 0.20  (1/5)
open_accuracy:   0.60  (3/5)
overall_accuracy: 0.40 (4/10)
avg_time_ms:     848.9
peak_vram_mb:    4716.4
```

Result file: `results/phase1_baseline/smolvlm2-2b_vqa_rad_seed42.json`

This confirms the entire evaluation pipeline works end-to-end:
- Dataset loading (VQA-RAD from HuggingFace)
- Question type classification (open/closed)
- Model inference
- Answer preprocessing and accuracy computation
- JSON result serialization

### 3.2 Florence-2-large on VQA-RAD (10 samples) - FAILED

All predictions were repetitive garbage text (e.g., "john john john...", "versvers Nuggets...").
Result file exists but all predictions are incorrect: `results/phase1_baseline/florence2-large_vqa_rad_seed42.json`

---

## 4. Florence-2 Debugging (Detailed)

### 4.1 Issues Found and Fixed

#### Issue 1: `forced_bos_token_id` AttributeError
- **Cause**: Florence-2's remote `configuration_florence2.py` accesses `self.forced_bos_token_id` before `super().__init__()` sets it. transformers 5.x raises AttributeError for unset attributes.
- **Fix**: `_florence2_compat()` context manager that temporarily patches `PretrainedConfig.__getattribute__` to return `None` for these specific attributes.
- **Location**: `src/baseline/model_loader.py:36-61`

#### Issue 2: Square image requirement
- **Error**: `"only support square feature maps"` from DaViT vision encoder
- **Fix**: Added automatic square resize in `_generate_direct_question()`:
  ```python
  if image.width != image.height:
      size = max(image.width, image.height)
      image = image.resize((size, size))
  ```
- **Location**: `src/baseline/model_loader.py:296-299`

#### Issue 3: Wrong prompt format
- **Cause**: Initially used `MEDICAL_PROMPT.format(question=question)` (free-form text). Florence-2 requires task-specific prefixes.
- **Fix**: Changed to `<VQA>` task prefix: `prompt = "<VQA>" + question`
- **Location**: `src/baseline/model_loader.py:292-293`

#### Issue 4: Missing embed_tokens / lm_head weight tying
- **Cause**: Checkpoint stores embeddings as `language_model.model.shared.weight` but transformers 5.x doesn't automatically tie `embed_tokens` and `lm_head` to the shared weight.
- **Fix**: Manual weight tying after loading:
  ```python
  lm.model.encoder.embed_tokens = lm.model.shared
  lm.model.decoder.embed_tokens = lm.model.shared
  lm.lm_head.weight = lm.model.shared.weight
  ```
- **Location**: `src/baseline/model_loader.py:133-139`

#### Issue 5: 192 all-zero bias parameters
- **Cause**: transformers 5.x meta-device initialization drops many bias parameters during weight materialization. These should have non-zero values (std ~0.7).
- **Fix**: Created `_florence2_fix_missing_weights()` that reloads zero-valued parameters from the safetensors checkpoint.
- **Location**: `src/baseline/model_loader.py:64-87`
- **Config**: `low_cpu_mem_usage=False` added to disable meta-device init (`model_loader.py:122-123`)

#### Issue 6: `post_process_generation()` for output parsing
- **Fix**: Florence-2 requires `processor.batch_decode(skip_special_tokens=False)` followed by `processor.post_process_generation(text, task="<VQA>", image_size=orig_size)` to extract the answer dict.
- **Location**: `src/baseline/model_loader.py:311-320`

### 4.2 Remaining Blocker: EncoderDecoderCache Incompatibility (Multi-session)

**Status: STILL UNRESOLVED after extensive patching across 2 sessions.**

- **Root cause**: Florence-2's BART-based decoder (custom `modeling_florence2.py`) expects `past_key_values` as old-style tuples: `((sa_key, sa_value, ca_key, ca_value), ...)`. transformers 5.x's `generate()` creates and manages an `EncoderDecoderCache` object wrapping two `DynamicCache` objects.
- **With `use_cache=False`**: Model generates garbage (confirmed with both float16 and float32). Autoregressive generation fundamentally requires KV cache to work correctly.
- **With `use_cache=True`**: `"EncoderDecoderCache object is not subscriptable"` error unless patched.

#### Session 2 Patches Applied (2026-03-21 ~ 2026-03-22)

All 4 patches applied to cached `modeling_florence2.py`:

**Patch 1 - Input conversion (line ~1855)**: Convert `EncoderDecoderCache → legacy tuple` at start of decoder forward.
- `DynamicLayer` initialization: `.keys` is `None` (from `CacheLayerMixin.__init__`), not missing attribute
- `sa_cache.layers[0].keys is None` correctly detects first decode step (pre-populated 12 empty layers)
- Converts to `((sa_k, sa_v, ca_k, ca_v), ...)` tuple for Florence-2's decoder

**Patch 2 - Output conversion (line ~1938)**: Convert decoder's tuple output `→ EncoderDecoderCache` for next step.
- Creates fresh `DynamicCache()` each step (avoids double-counting)
- `DynamicCache.update(key, value, layer_idx)` signature: layer_idx is 3rd positional arg

**Patch 3 - `_reorder_cache` (line ~2259)**: Handle `EncoderDecoderCache.reorder_cache(beam_idx)` for beam search.

**Patch 4 - `generate()` attention_mask (line ~2832)**: Override text-only `attention_mask (1, 13)` with merged `attention_mask (1, 590)` after `_merge_input_ids_with_image_features()`.
- Image processor: 768x768 → 577 image tokens + 13 text tokens = 590 total
- Without fix: encoder sees shape mismatch; with fix: encoder sees full 590-token sequence

#### Garbage Output After All Patches

After all 4 patches, model still produces garbage:
- Without attention_mask fix (earlier): `"begg begg begg begg..."` (128 repetitions)
- With attention_mask fix (current): `"</s><s><s><s>..."` - generates EOS then BOS repeatedly

The output TYPE changed (different degenerate patterns), confirming each patch has SOME effect, but the model cannot produce meaningful text.

#### Session 3 Final Diagnosis: SA Cache Completely Non-Functional (2026-03-22)

**Test**: 3-step manual forward pass (`tests/florence2_step_test.py`) calling `lm.model.decoder()` directly with explicit cache passing.

**Result**:
```
Step 1 top-5 (input: token 2 = </s>): <s> 71.78%, V 26.42%, Q 1.54%
Step 2 top-5 (input: token 0 = <s>):  <s> 71.78%, V 26.42%, Q 1.51%  ← IDENTICAL
Step 3 top-5 (input: token 0 = <s>):  <s> 71.78%, V 26.42%, Q 1.51%  ← IDENTICAL
SA cache populated: 12 layers, layer[0].keys shape [1, 16, 1, 64]
```

**Root cause confirmed**: The SA cache IS being populated and converted by Patch 1↔2, but the decoder's internal causal attention mask for SA is computed based only on the current input length (shape `[1,1,1,1]`), not the total decoded length (shape `[1,1,1,cache+1]`). This masks out all cached SA keys, making each step's output identical to the first — the model only attends to encoder cross-attention, giving the same distribution every time.

**Required 5th patch**: Extend decoder SA attention mask to cover cached positions. This would require patching `Florence2BartDecoder.forward()` to pass the correct `decoder_attention_mask` that covers all previously decoded positions when `past_key_values` is provided.

**Decision: ABANDON Florence-2 debugging. Drop from evaluation.**

Evidence supporting drop:
1. SA cache entirely non-functional after 3 sessions + 4 patches
2. Requires a 5th patch with high risk of revealing further issues
3. Model is smallest (0.77B) with lowest expected contribution to thesis
4. 3 working models sufficient for comparative analysis

#### DynamicCache API (transformers 5.1.0) - Key Finding

`DynamicLayer` in transformers 5.1.0:
- `CacheLayerMixin.__init__`: sets `self.keys = None`, `self.values = None`, `self.is_initialized = False`
- `lazy_initialization()`: sets `self.keys = torch.tensor([], ...)` (empty tensor), then `is_initialized = True`
- `update(key_states, value_states)`: appends via `torch.cat([self.keys, key_states], dim=-2)`
- `DynamicCache.update(key, value, layer_idx)`: correct API used in Patch 2
- Pre-populated layers: `generate()` creates `DynamicCache(config=model.config)` pre-allocating 12 layers (for 12 decoder layers), all with `.keys = None` initially

**Key files involved**:
- Cached model: `~/.cache/huggingface/modules/transformers_modules/microsoft/Florence_hyphen_2_hyphen_large/.../modeling_florence2.py`
  - Decoder forward: line 1855 (Patch 1: input conversion)
  - Decoder forward: line 1938 (Patch 2: output conversion)
  - `_reorder_cache`: line 2259 (Patch 3: beam search)
  - `Florence2ForConditionalGeneration.generate()`: line 2832 (Patch 4: attention_mask)
- transformers 5.1.0 cache: `.venv/Lib/site-packages/transformers/cache_utils.py`
- transformers 5.1.0 BART: `.venv/Lib/site-packages/transformers/models/bart/modeling_bart.py`
  - Key insight: Standard BART uses `is_updated` flag in `EncoderDecoderCache` for CA reuse; Florence-2's custom attention does NOT use this flag

### 4.3 Decision: Drop Florence-2

**FINAL DECISION (2026-03-22): Drop Florence-2 from Phase 1 evaluation.**

Root cause fully diagnosed after 3 sessions:
- SA causal attention mask not extended for cached positions (5th patch required)
- Each additional patch revealed deeper incompatibilities
- Not worth continued effort given model size (0.77B) and thesis scope

**Thesis treatment**: Document as a technical finding — "Florence-2 (microsoft/Florence-2-large) was excluded due to verified incompatibility with transformers≥5.0 arising from EncoderDecoderCache wrapper introduced in transformers 5.x, which breaks the old-style tuple cache assumed by Florence-2's custom BART decoder implementation."

Alternative options (not pursued):
1. Pin `transformers==4.47.1` in separate env — viable but disrupts project env
2. Wait for Microsoft upstream fix — no timeline known

---

## 5. Implemented Files

### 5.1 Model Configs (`configs/models/`)

| File | Model | Key Settings |
|------|-------|-------------|
| `qwen3_vl_2b.yaml` | Qwen3-VL-2B | float16, device_map=auto, **sdpa**, max_pixels=401408, chat_template, requires_vision_info |
| `qwen25_vl_3b.yaml` | Qwen2.5-VL-3B | float16, device_map=auto, **sdpa**, max_pixels=401408, chat_template, requires_vision_info |
| `florence2_large.yaml` | Florence-2-large | float16, device_map=null, direct_question, trust_remote_code, eager attn, num_beams=3 |
| `smolvlm2_2b.yaml` | SmolVLM2-2.2B | bfloat16, device_map=auto, chat_template |

### 5.2 Core Source Files

| File | Purpose |
|------|---------|
| `src/baseline/model_loader.py` | Model loading + single/batch inference (`load_model`, `generate_answer`, `generate_answers_batch`) |
| `src/baseline/evaluate_zero_shot.py` | Evaluation engine (`evaluate_with_loaded_model`, `_infer_batch`, `_infer_single`) |
| `src/baseline/run_all.py` | All conditions runner — model loaded ONCE per config, shared across all dataset+seed combos |
| `src/data/download.py` | HuggingFace dataset downloader |
| `src/data/dataset.py` | Unified VQASample dataclass and dataset loading |
| `src/evaluate/metrics.py` | VQA accuracy metrics (closed/open/overall) |
| `src/utils/seed.py` | Deterministic seeding |
| `src/utils/vram_monitor.py` | GPU VRAM monitoring |

### 5.3 Inference Path per Model

```
Single inference:
  Qwen3-VL-2B / Qwen2.5-VL-3B:
    generate_answer -> _generate_chat_template -> _generate_qwen_style
    (uses qwen_vl_utils.process_vision_info)

  SmolVLM2-2.2B:
    generate_answer -> _generate_chat_template -> _generate_standard_chat
    (two-step: apply_chat_template -> processor)

  Florence-2-large:
    generate_answer -> _generate_direct_question
    (uses <VQA> task prefix + post_process_generation)

Batch inference (batch_size > 1):
  Qwen3-VL-2B / Qwen2.5-VL-3B:
    generate_answers_batch -> _generate_qwen_style_batch
    (per-sample process_vision_info, left-pad tokenizer, batch generate)

  SmolVLM2-2.2B:
    generate_answers_batch -> _generate_standard_chat_batch
    (images as [[img] for img in images] — Idefics3 nested list requirement)

  Florence-2-large:
    generate_answers_batch -> falls back to single inference (direct_question not batched)
```

---

## 6. Performance Optimizations Applied (2026-03-22)

Five bottleneck fixes applied to reduce Phase 1 total runtime from ~12-15h to ~1.5-2h (seed=42 pass).

| Fix | Change | Effect |
|-----|--------|--------|
| **Batch inference** | `batch_size=4` in `run_phase1.bat` | 2-4x throughput |
| **SDPA attention** | `attn_implementation: "sdpa"` in Qwen configs | ~20-30% faster |
| **Reduced max_pixels** | 802816 → 401408 in Qwen configs | ~40-50% faster image processing |
| **Model load once** | Load model ONCE per config, share across 9 conditions | Eliminates 30-60s × 8 repeated loads |
| **Single seed first** | `--single_seed_first` flag, run seed=42 only | ~3x time reduction for representative pass |

**CLI flags added to `run_all.py`**:
- `--batch_size INT` (default: 4)
- `--single_seed_first` (run seeds[0] only, add remaining seeds later)
- `--torch_compile` (optional ~15-30% more speedup via `torch.compile`)

**flash_attn attempt (abandoned)**:
- Downloaded `flash_attn-2.8.3+cu129sm120-cp312-cp312-win_amd64.whl` (White2Hand HuggingFace)
- Installed successfully after Python 3.13→3.12 venv downgrade
- Failed at runtime: `DLL load failed` — cu129 binary incompatible with torch cu128 runtime
- Decision: Keep SDPA (already provides good speedup, no DLL issues)

**SmolVLM2 batch format bug fixed**:
- Error: `"The number of images in the text [1,1,1,1] and images [4] should be the same"`
- Root cause: Idefics3-based processor interprets flat `images=[img1,img2,img3,img4]` as 4 images for ONE sample
- Fix: `images=[[img] for img in images]` (nested list, one list per sample)

---

## 7. Next Steps

### Florence-2: CLOSED (2026-03-22)
- [x] Decision made: **Drop Florence-2** — SA cache non-functional, 5th patch required
- Evaluation proceeds with 3 models: Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B

### Completed
- [x] Run 10-sample test with all 3 models (all 9 conditions verified)
- [x] Add `num2words` to `pyproject.toml`
- [x] Apply 5 performance optimizations (batch, SDPA, max_pixels, model-once, single_seed)
- [x] Python venv downgraded 3.13 → 3.12

### In Progress
- [🔄] **Phase 1 seed=42 pass running** (`run_phase1.bat`, started 2026-03-22 15:28)
  - 9 conditions: Qwen3-VL-2B + Qwen2.5-VL-3B + SmolVLM2-2.2B × PathVQA + SLAKE + VQA-RAD
  - Log: `results/phase1_baseline/run_all.log`

### After seed=42 completes
- [ ] Check results: verify accuracy and VRAM per model/dataset
- [ ] Run seeds 123 + 456: remove `--single_seed_first` from `run_phase1.bat`, re-run
- [ ] Generate final `results/phase1_baseline/phase1_summary.csv`

### Datasets
| Dataset | HF ID | Test Size |
|---------|-------|-----------|
| PathVQA | `flaviagiammarino/path-vqa` | ~6,719 QA |
| SLAKE | `BoKelvin/SLAKE` (filter `q_lang=="en"`) | ~1,044 QA |
| VQA-RAD | `flaviagiammarino/vqa-rad` | ~451 QA |

---

## 7. Key Learnings

1. **transformers 5.x breaking changes**: The `EncoderDecoderCache` wrapper is a fundamental architecture change. Models with custom `modeling_*.py` that expect old-style tuple caches (like Florence-2's BART decoder) will break.

2. **Weight tying is not automatic**: When models use `trust_remote_code=True`, transformers 5.x may not properly tie shared embeddings. Manual tying (`embed_tokens = shared`, `lm_head.weight = shared.weight`) may be needed.

3. **Meta-device init drops parameters**: `low_cpu_mem_usage=True` (default) uses meta tensors that can silently drop bias parameters during materialization. Set `low_cpu_mem_usage=False` for models with known issues.

4. **SmolVLM2 chat template quirk**: Unlike most models, SmolVLM2's `apply_chat_template` does NOT accept `images` parameter. Must use two-step approach: get text first, then pass to `processor()` with images.

5. **Florence-2 requires square images**: DaViT vision encoder only supports square feature maps. Non-square images must be resized before processing.

6. **SmolVLM2 batch format (Idefics3)**: Flat `images=[img1, img2, img3, img4]` is misinterpreted as 4 images for one sample. Must use nested list: `images=[[img] for img in images]` — one list per sample.

7. **Python version matters for prebuilt wheels**: flash_attn prebuilt wheels for sm_120 (Blackwell) only exist for cp312. Python 3.13 (cp313) venv required a downgrade to 3.12 before installation. Even then, cu129-compiled flash_attn DLLs are binary-incompatible with torch cu128 runtime — SDPA is the practical choice on this setup.

8. **uv venv vs pip**: `uv venv` does not install pip by default. Use `uv pip install` instead of `.venv/Scripts/python.exe -m pip install`.
