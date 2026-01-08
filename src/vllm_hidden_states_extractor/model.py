# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionBackend, AttentionType
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.attention.selector import get_attn_backend
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.multimodal.inputs import NestedTensors

from vllm.model_executor.models.utils import maybe_prefix
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

logger = init_logger(__name__)


# @support_torch_compile(
#     dynamic_arg_dims={
#         "input_ids": 0,
#         "positions": -1,
#         "hidden_states": 0,
#         "input_embeds": 0,
#     }
# )


class DummyModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(self.config, "eagle_aux_hidden_state_layer_ids", [])
        )

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size // self.num_hidden_states,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )


class CacheOnlyAttentionLayer(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str, config, num_kv_heads: int):
        super().__init__()

        vllm_config = get_current_vllm_config()
        cache_config = vllm_config.cache_config
        self.block_size = cache_config.block_size

        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, vllm_config.model_config
        )

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.attn_backend = get_attn_backend(
            head_dim,
            torch.get_default_dtype(),
            cache_config.cache_dtype,
            self.block_size,
            use_mla=False,
            has_sink=False,
            use_mm_prefix=False,
            attn_type=AttentionType.DECODER,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
        )


class HiddenStatesExtractor(Eagle3LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        print("HiddenStatesExtractor __init__")
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(self.config, "eagle_aux_hidden_state_layer_ids", [])
        )
        # Ensure draft_vocab_size is set
        # default to the base vocab size when absent
        if getattr(self.config, "draft_vocab_size", None) is None:
            base_vocab_size = getattr(self.config, "vocab_size", None)
            self.config.draft_vocab_size = base_vocab_size
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )

        # Store target layer count in draft config for
        # proper layer_types indexing in draft models
        self.config.target_layer_count = target_layer_num

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size // self.num_hidden_states,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

        self.layers = nn.ModuleList(
            [
                CacheOnlyAttentionLayer(
                    maybe_prefix(prefix, f"layers.{layer_idx + target_layer_num}"),
                    self.config,
                    getattr(
                        self.config,
                        "num_key_value_heads",
                        self.config.num_attention_heads,
                    ),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.model = DummyModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # todo: Cache hidden states
        # hidden_states is (batch_size, hidden_size * num_hidden_states)

        # forward_context: ForwardContext = get_forward_context()
        # attn_metadata = forward_context.attn_metadata
        # self_kv_cache = self.layers[0].kv_cache[forward_context.virtual_engine]

        dummy_ret = hidden_states.new_zeros(
            (hidden_states.shape[0], hidden_states.shape[1] // self.num_hidden_states)
        )
        return dummy_ret, dummy_ret

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, (
                "Expected logits to have shape "
                f"(*, {self.config.vocab_size}), but got {logits.shape}"
            )
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (
                logits.shape[0],
                self.config.vocab_size,
            ),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # no-op. We return the full hidden states so that they are all passed into the forward fn where they will be cached.
        # Note: this requires setting the dummy model hidden size to (num_hidden_states * hidden_size)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return None
