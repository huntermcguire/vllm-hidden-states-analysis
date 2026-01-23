# Config for Qwen3-8B hidden states extractor

Based on Eagle3 config from RedHatAI/Qwen3-8B-speculator.eagle3.

Modifications:
- changed the "speculators_model_type" field to "extract_hidden_states"
- set the "eagle_aux_hidden_state_layer_ids" field to the layer ids of the hidden layers to extract. For PoC, there must be an even number of layers. 
- set the "transformer_layer_config.hidden_size" field to the original hidden size * the number of hidden states to extract. (e.g. `4096 * 4 = 16384` or the first 4 layers)
- set "transformer_layer_config.num_key_value_heads" to the same value as "transformer_layer_config.num_attention_heads"

Note: eventually it should be possible to do this automatically. 

Currently using the cli or python api directly (without creating this config), triggers a validation check that fails because "extract_hidden_states" is not a valid speculator type. By creating a file config and loading it, that check is bypassed.

The "transformer_layer_config.hidden_size" is set to that multiple to bypass an assert in gpu_model_runner which checks the shape of the hidden states after they are combined. Since we want to get all the hidden states, we replace the combine op (defined on the dummy speculator) with an identity op and then change the hidden size. This could be fixed with minor changes to the vLLM code.

Setting "transformer_layer_config.num_key_value_heads" could also be skipped by just improving the dummy speculator logic to automatically determint the num heads from hidden size and head dim.