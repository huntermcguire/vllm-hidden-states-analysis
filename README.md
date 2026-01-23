# vllm-hidden-states-extractor
WIP Plugin for extracting hidden states from vLLM.

## Usage
Current usage during experimentation is:

1. Install vllm and this plugin:
```bash
uv pip install -e . 
```
Note: vLLM 0.14.0 is a dependency of the plugin and will be installed automatically.

2. Serve the model with kv connector
```bash
vllm serve ./demo/qwen3_8b --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config": {"shared_storage_path": "/tmp/hidden_states"}}'

For more information on the model config, see `demo/qwen3_8b/README.md`.
```
3. Send a request to the model:
```bash
curl http://localhost:8000/v1/completions     -H "Content-Type: application/json"     -d '{
        "model": "./demo/qwen3_8b",
        "prompt": "Why are hidden states required for Eagle3 training?"
    }'
```
4. Verify the hidden states are extracted.
These will be saved to `/tmp/hidden_states/{request_id}/hidden_states.safetensors`. For this PoC
they are stored as a safetensors file with two tensors: 
  - "hidden_states": [num_hidden_states=4, seq_len, hidden_size] 
  - "token_ids": [seq_len]


## Structure
In `pyproject.toml`, the plugin is registered
```toml
[project.entry-points."vllm.general_plugins"]
register_hidden_states_extractor = "vllm_hidden_states_extractor:register"
```

When vLLM is initialized, it will call the `register` function in `src/vllm_hidden_states_extractor/__init__.py`, which initializes the plugin.

In `src/vllm_hidden_states_extractor/__init__.py`, the `register` function registers the "HiddenStatesExtractor" model and a fake speculator type "extract_hidden_states" (with its handler function) and the "ExampleHiddenStatesConnector" kv connector.

In `src/vllm_hidden_states_extractor/model.py`, the `HiddenStatesExtractor` model is defined. It is intended to be a dummy model that just caches the received hidden states into its layers "KV cache".

In `src/vllm_hidden_states_extractor/attention.py`, the `CacheOnlyAttentionBackend` is defined. It is a custom attention backend that just caches the received hidden states into its layers "KV cache".

In `src/vllm_hidden_states_extractor/model.py`, the `CacheOnlyAttentionLayer` is defined. It is a custom attention layer intended to work with the `CacheOnlyAttentionBackend`. This is partially needed because otherwise FDSP has a check that finds all `Attention` (official vllm attention class) layers and checks that they are using the FSDP backend. Unfortunately, this will fail for our custom attention backend. By creating a custom attention layer that also subclasses `AttentionLayerBase`, we can bypass this check.

In `src/vllm_hidden_states_extractor/connector.py`, the `ExampleHiddenStatesConnector` is defined. It is a simple kv connector that extracts the kv cache for each request (only from CacheOnlyAttentionLayers), reshapes the layers to match the hidden states shape, and saves them to disk. 


## Status

- [x] Add plugin registration to pyproject.toml
- [x] Register dummy model and fake speculator
- [x] Implement a dummy model placeholder
- [x] Handle logic to prevent multiple hidden states getting combined before model forward is called
- [x] Cache hidden states received by the model into its layers "KV cache"
- [x] Use existing KVCacheConnector to extract all hidden states (Needed to modify the connector)
- ~~[ ] Create a filter KVConnector to only extract hidden states from dummy layers~~
- ~~[ ] Cleanup model code~~
- [x] Create a simple KV connector that writes hidden states to disk
- [ ] Determine what vLLM changes are needed for better support