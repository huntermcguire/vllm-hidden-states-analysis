# vllm-hidden-states-analysis

Fork of [vllm-hidden-states-extractor](https://github.com/vllm-project/vllm-hidden-states-extractor) modified for live classification of hidden states at runtime.

## Changes from upstream

This fork modifies the hidden states extraction to:

1. **Live classification instead of disk writes** — Hidden states are passed to a pre-trained classifier at runtime rather than saved to disk.
2. **Last token only** — Only extracts hidden states from the last token of the prompt during first-pass prefill.
3. **Probe scores in response** — Classification results are returned in the API response via `kv_transfer_params.probe_scores`.

## How it works

The plugin hijacks vLLM's Eagle3 speculative decoding support to intercept hidden states from configurable layers. During prefill, the connector extracts the last token's hidden state and runs it through per-layer sklearn probes (PCA → StandardScaler → LogisticRegression) to produce a refusal probability score.

## Usage

1. Install:
```bash
uv pip install -e .
```

2. Serve the model with a classifier directory:
```bash
vllm serve ./demo/qwen3_8b --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config": {"classifier_dir": "/path/to/classifier_models"}}'
```

3. Send a request:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "./demo/qwen3_8b", "prompt": "Your prompt here"}'
```

The response will include `probe_scores` with per-layer refusal probabilities.

## Classifier format

Classifiers are discovered by model name. Place a `layer_probes.pkl` file at:
```
{classifier_dir}/{model_name}/layer_probes.pkl
```

Where `model_name` has `/` replaced with `--` (e.g., `meta-llama/Llama-3.1-8B-Instruct` becomes `meta-llama--Llama-3.1-8B-Instruct`).

The pickle file should contain:
```python
{
    "probes": {layer_idx: {"pca": ..., "scaler": ..., "lr": ...}, ...},
    "probe_layers": [layer_indices...]
}
```