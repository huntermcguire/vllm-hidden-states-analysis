# vllm-hidden-states-extractor
WIP Plugin for extracting hidden states from vLLM.

## Usage
Current usage during experimentation is:

1. pip install vllm and this plugin project. (`pip install .`)
2. Download an eagle3 spec model from the Hugging Face Model Hub. e.g. 
```bash
hf download RedHatAI/Qwen3-8B-speculator.eagle3 --local-dir Qwen3-8B-speculator.eagle3
```
3. Manually change the "speculators_model_type" field to "extract_hidden_states" in the downloaded models `config.json` file.
4. Load the modified model with vLLM:
```python
from vllm import LLM
llm = LLM(model="Qwen3-8B-speculator-eagle3")
llm.generate("Hello, world!")
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


## Status

- [x] Add plugin registration to pyproject.toml
- [x] Register dummy model and fake speculator
- [x] Implement a dummy model placeholder
- [ ] Handle logic to prevent multiple layers getting combined before model forward is called
- [ ] Cache hidden states received by the model into its layers "KV cache"
- [ ] Use existing KVCacheConnector to extract all hidden states
- [ ] Create a filter KVCacheConnector to only extract hidden states from dummy layers
- [ ] Cleanup model code
- [ ] Swap out vLLM proposer class for simpler one?
- [ ] Determine what vLLM changes are needed for better support