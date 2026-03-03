# Plan: Live Jailbreak Classification via Hidden States

## Goal

Modify the vllm-hidden-states-extractor plugin so that instead of saving hidden states to disk, it extracts hidden states from 9 sampled layers for the last token of the first prefill forward pass and runs them against per-layer sklearn probes to determine jailbreak probability live. All 9 per-layer scores are reported individually (no aggregation). Abort mechanism is deferred to a later phase.

## Current Architecture

1. A fake Eagle3 speculator config lists `eagle_aux_hidden_state_layer_ids` — vLLM's existing plumbing concatenates hidden states from those layers and passes them to the dummy model
2. The dummy model (`HiddenStatesExtractor`) stores the concatenated hidden states into a fake KV cache via `CacheOnlyAttentionLayer`
3. The `ExampleHiddenStatesConnector` extracts hidden states from the KV cache and saves them to disk as safetensors

## Target Model

- **Model**: `meta-llama/Llama-3.1-8B-Instruct` (32 hidden layers, hidden_size=4096, head_dim=128)
- **Layers to extract**: `[0, 4, 8, 12, 16, 20, 24, 28, 32]` — every 4th layer, 9 total
- **Matches**: `probe_layers` in the trained classifier pkl exactly
- **K/V hack math**: 9 * 4096 / (2 * 128) = 144 — clean integer, no duplication needed
- **Speculator hidden_size**: 9 * 4096 = 36864

## Classifier Structure

Each of the 9 layers has an independent sklearn probe pipeline:

1. **PCA** (4096 → 32 dimensions)
2. **StandardScaler** (normalize the 32 PCA features)
3. **LogisticRegression** (32 features → binary refuse/comply)

Input: last token's hidden state at a given layer — shape `[4096]`
Output: refuse probability (float)

The probes are stored in a single pkl file containing a `probes` dict (layer_idx → pipeline) and a `probe_layers` list.

## What Needs to Change

### 1. Config: Llama-8B speculator config

- Create `classifier_models/meta-llama--Llama-3.1-8B-Instruct/config.json` with:
  - `eagle_aux_hidden_state_layer_ids`: `[0, 4, 8, 12, 16, 20, 24, 28, 32]`
  - `hidden_size`: 36864 (9 * 4096)
  - `num_attention_heads` and `num_key_value_heads`: 144 (36864 / (2 * 128))
  - `head_dim`: 128
  - `vocab_size` and `draft_vocab_size` matching Llama-3.1-8B
  - `verifier.name_or_path`: `meta-llama/Llama-3.1-8B-Instruct`
- The config lives alongside the classifier pkl in the same model directory

### 2. Optimize to last-token-only extraction

- In `CacheOnlyAttentionLayer.forward()` in `model.py`:
  - Retrieve `attn_metadata` via `get_forward_context()` (the pattern exists in commented-out code already)
  - Compute last-token indices from `attn_metadata.query_start_loc`: index `query_start_loc[i+1] - 1` for each request
  - Filter `hidden_states` to just those rows (1 per request)
  - Filter `attn_metadata.slot_mapping` to the same indices — both must be filtered in sync before calling into the attention backend, otherwise cache writes go to wrong slots
  - The existing `cache_only_attention_with_kv_transfer` and `reshape_and_cache_flash` then operate on 1 token per request with no code changes needed in `attention.py`
- The connector's `save_kv_layer` then extracts just that single slot per request
- This dramatically reduces memory usage and simplifies the classifier input

### 3. Replace disk save with live classifier

- **New file**: `src/vllm_hidden_states_extractor/classifier.py`
  - vLLM-agnostic module — takes numpy arrays (CPU), returns scores
  - Loads the pkl probes once at init
  - Runs each layer's probe: hidden_state[4096] → PCA[→32] → StandardScaler → LogisticRegression → refuse_prob
  - Returns all 9 per-layer scores as a dict (layer_idx → refuse_probability)

- **Dynamic classifier discovery**:
  - Connector gets model name from `vllm_config.model_config` at init
  - Sanitizes the name (`/` → `--`)
  - Looks up `{classifier_dir}/{sanitized_name}/layer_probes.pkl`
  - `classifier_dir` is passed via `kv_connector_extra_config`
  - If no classifier found: **skip classification entirely and log a warning** (no fallback to disk save)
  - `probe_layers` in the pkl self-describes which layers the classifier expects — must match `eagle_aux_hidden_state_layer_ids` in the config

- **Connector changes** (`connector.py`):
  - Replace safetensors disk save in `save_kv_layer` with a call to the classifier
  - Hidden states come off GPU as tensors — `.detach().cpu().numpy()` before passing to classifier (existing pattern from safetensors save)
  - Store scores in `self._request_scores[req_id]` dict (same pattern as existing `self._request_filenames`) so `request_finished()` can retrieve them later
  - `request_finished()` returns all 9 probe scores via KV transfer params (replacing the file path)
  - Log all 9 scores per request for monitoring
  - Remove `self._request_filenames`, `self._storage_path`, safetensors import, and filename logic from `ReqMeta` / `build_connector_meta`

### 4. Abort mechanism (deferred)

Deferred to a later phase. For now, classification scores are reported but generation is not interrupted.

## Feasibility Assessment

| Requirement | Difficulty | Notes |
|---|---|---|
| Llama config | Low | New config.json with correct layer IDs and sizing |
| Last token only | Low | Use `query_start_loc` to filter in `CacheOnlyAttentionLayer.forward()` |
| First prefill only | Already done | `build_connector_meta` only fires for `scheduled_new_reqs` |
| Live classifier | Low | sklearn probes are tiny — PCA + scaler + logistic regression |
| Dynamic discovery | Low | Model name lookup + pkl load |
| Abort on detection | Deferred | Will require a callback/signal mechanism |

## Key Concerns

- **Memory**: 9-layer extraction is modest. Last-token-only optimization means the dummy KV cache stores just 1 slot per request
- **Latency**: sklearn probes (PCA + scaler + logistic regression) are negligible — microseconds per layer
- **K/V divisibility**: The constraint is that `num_layers * hidden_size` must be divisible by `2 * head_dim`. For Llama-8B with 9 layers: 9 * 4096 / (2 * 128) = 144. Clean integer, no issues
- **Config/classifier sync**: `eagle_aux_hidden_state_layer_ids` in config.json must match `probe_layers` in the pkl. Both live in the same directory to make this obvious

## File Structure

```
src/vllm_hidden_states_extractor/
  __init__.py          # plugin registration (unchanged)
  attention.py         # CacheOnlyAttentionBackend (unchanged)
  classifier.py        # NEW — loads pkl probes, runs PCA→scaler→LR per layer
  connector.py         # replace disk save → classifier call, return scores
  model.py             # last-token-only caching in CacheOnlyAttentionLayer.forward()
  utils.py             # reshape helpers (unchanged)

classifier_models/
  meta-llama--Llama-3.1-8B-Instruct/
    config.json        # speculator config for this model
    layer_probes.pkl   # trained sklearn probes

llama_layer_probe_ref.py   # reference implementation (stays at repo root)
```

## Files to Modify

- `src/vllm_hidden_states_extractor/connector.py` — replace disk save with classifier call, dynamic discovery, return scores
- `src/vllm_hidden_states_extractor/model.py` — last-token-only caching via `query_start_loc` filtering
- `src/vllm_hidden_states_extractor/classifier.py` — **new file**, vLLM-agnostic probe runner

## Files to Create

- `classifier_models/meta-llama--Llama-3.1-8B-Instruct/config.json` — Llama speculator config
- `classifier_models/meta-llama--Llama-3.1-8B-Instruct/layer_probes.pkl` — moved from repo root

## Dependencies to Add

- `scikit-learn` in `pyproject.toml` — required to unpickle and run the sklearn probe pipelines (PCA, StandardScaler, LogisticRegression)

## Files to Remove

- `demo/` — entire directory (from forked project, built around old save-to-disk flow)
- `layer_probes_llama.pkl` — moved into `classifier_models/` subdirectory
- `llama_layer_probe_ref.py` - remove after using as reference material for `src/vllm_hidden_states_extractor/classifier.py`
