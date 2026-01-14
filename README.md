# vllm-hidden-states-extractor
WIP Plugin for extracting hidden states from vLLM.

## Usage
Current usage during experimentation is:

1. pip install vllm and this plugin project. (`pip install .`)
2. Download an eagle3 spec model from the Hugging Face Model Hub. e.g. 
```bash
hf download RedHatAI/Qwen3-8B-speculator.eagle3 --local-dir Qwen3-8B-speculator.eagle3
```
3. Manually update `config.json` file:
- change the "speculators_model_type" field to "extract_hidden_states"
- set the "eagle_aux_hidden_state_layer_ids" field to the layer ids of the hidden states to extract. (e.g. `[1, 2, 3, 4]` for the first 4 layers)
- set the "transformer_layer_config.hidden_size" field to the original hidden size * the number of hidden states to extract. (e.g. `1024 * 4 = 4096` for the first 4 layers)
4. Load the modified model with vLLM:
```python
from vllm import LLM

from vllm.config import KVTransferConfig

ktc = KVTransferConfig(
    kv_connector="ExampleConnector",
    kv_role="kv_producer",
)

llm = LLM(model="Qwen3-8B-speculator-eagle3", kv_transfer_config=ktc)
outputs = llm.generate("Hello world")

print(outputs)
```

## Structure
In `pyproject.toml`, the plugin is registered as a "plugin" using:
```toml
[project.entry-points."vllm.general_plugins"]
register_hidden_states_extractor = "vllm_hidden_states_extractor:register"
```

When vLLM is initialized, it will call the `register` function in `src/vllm_hidden_states_extractor/__init__.py`, which registers the plugin.

In `src/vllm_hidden_states_extractor/__init__.py`, the `register` function registers the "HiddenStatesExtractor" model and a fake speculator type "extract_hidden_states" (with its handler function).

In `src/vllm_hidden_states_extractor/model.py`, the `HiddenStatesExtractor` model is defined. It is intended to be a dummy model that just caches the received hidden states into its layers "KV cache".

In `src/vllm_hidden_states_extractor/attention.py`, the `CacheOnlyAttentionBackend` is defined. It is a custom attention backend that just caches the received hidden states into its layers "KV cache".

In `src/vllm_hidden_states_extractor/model.py`, the `CacheOnlyAttentionLayer` is defined. It is a custom attention layer intended to work with the `CacheOnlyAttentionBackend`. This is partially needed because otherwise FDSP has a check that finds all `Attention` (official vllm attention class) layers and checks that they are using the FSDP backend. Unfortunately, this will fail for our custom attention backend. By creating a custom attention layer that also subclasses `AttentionLayerBase`, we can bypass this check.


## Status

- [x] Add plugin registration to pyproject.toml
- [x] Register dummy model and fake speculator
- [x] Implement a dummy model placeholder
- [x] Handle logic to prevent multiple hidden states getting combined before model forward is called
- [x] Cache hidden states received by the model into its layers "KV cache"
- [ ] Use existing KVCacheConnector to extract all hidden states
- [ ] Create a filter KVCacheConnector to only extract hidden states from dummy layers
- [ ] Cleanup model code
- [ ] Swap out vLLM proposer class for simpler one?
- [ ] Determine what vLLM changes are needed for better support