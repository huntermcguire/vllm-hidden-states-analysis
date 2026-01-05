# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionBackend, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.attention.selector import get_attn_backend
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import SupportsEagle3
from vllm.multimodal.inputs import NestedTensors

from vllm.model_executor.models.utils import (
    get_draft_quant_config,
    maybe_prefix,
)
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

logger = init_logger(__name__)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "hidden_states": 0,
        "input_embeds": 0,
    }
)
class LlamaModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size

        # Get drafter's quantization config
        self.quant_config = get_draft_quant_config(vllm_config)

        eagle_config = getattr(self.config, "eagle_config", None)
        if eagle_config is not None and "use_aux_hidden_state" in eagle_config:
            self.use_aux_hidden_state = eagle_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        if self.use_aux_hidden_state:
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * 3
            else:
                fc_input_size = self.config.hidden_size * 3
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == input_embeds.shape[-1]

        residual = None
        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
        return hidden_states, hidden_prenorm

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            # Handle kv cache quantization scales
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            # Remapping the name FP8 kv-scale
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


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


class HiddenStatesExtractor(nn.Module, SupportsEagle3):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        print("HiddenStatesExtractor __init__")
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
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
        self.model = LlamaModel(
            vllm_config=vllm_config, prefix="model", start_layer_id=target_layer_num
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
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

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(hidden_states), torch.empty_like(hidden_states)

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
        if not self.model.use_aux_hidden_state:
            return hidden_states
        # combine multiple auxiliary hidden states returned by eagle3
        return self.model.fc(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return None
