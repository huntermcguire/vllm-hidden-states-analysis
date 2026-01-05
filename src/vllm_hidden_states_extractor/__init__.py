
def register():
    from vllm import ModelRegistry

    if "HiddenStatesExtractor" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("HiddenStatesExtractor", "vllm_hidden_states_extractor.model:HiddenStatesExtractor")

