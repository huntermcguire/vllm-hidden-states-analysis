# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.v1.attention.backend import AttentionMetadata
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_hidden_states_extractor.classifier import HiddenStateClassifier
from vllm_hidden_states_extractor.model import CacheOnlyAttentionLayer
from vllm_hidden_states_extractor.utils import reshape_hidden_states_from_kv_cache

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Shared across connector instances (GPU worker writes, scheduler reads).
# Both instances live in the same engine process so a module-level dict works.
_shared_scores: dict[str, dict[int, float]] = {}


@dataclass
class ReqMeta:
    req_id: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor

    @staticmethod
    def make_meta(
        req_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> "ReqMeta":
        token_ids_tensor = torch.tensor(token_ids)
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()
        return ReqMeta(
            req_id=req_id,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
        )


@dataclass
class ExampleHiddenStatesConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        req_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(req_id, token_ids, block_ids, block_size)
        )


class ExampleHiddenStatesConnector(KVConnectorBase_V1):

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self.cache_layers: list[str] = []

        spec_config = self._vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(spec_config, "eagle_aux_hidden_state_layer_ids", [])
        )

        # --- Classifier discovery ---
        classifier_dir = self._kv_transfer_config.get_from_extra_config(
            "classifier_dir", None
        )
        model_name = vllm_config.model_config.model
        self._classifier: HiddenStateClassifier | None = None

        if classifier_dir and model_name:
            self._classifier = HiddenStateClassifier.discover(
                classifier_dir, model_name
            )
        else:
            logger.warning(
                "classifier_dir not set in kv_connector_extra_config. "
                "Classification will be skipped."
            )

        # Scores are stored in module-level _shared_scores dict
        # because vLLM creates two connector instances (GPU worker + scheduler).

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, kv_caches.keys()
        )
        self.cache_layers = list(layers.keys())
        logger.info(f"Found {len(self.cache_layers)} CacheOnlyAttentionLayers")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if layer_name not in self.cache_layers:
            return

        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
            num_tokens: int,
        ) -> torch.Tensor:
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            padded_kv = layer.reshape(2, num_pages * page_size, -1)[
                :, slot_mapping, ...
            ]
            return padded_kv[:, :num_tokens, ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, ExampleHiddenStatesConnectorMetadata)

        for request in connector_metadata.requests:
            # Extract only the last token's slot from the KV cache.
            last_pos = request.token_ids.shape[0] - 1
            last_slot = request.slot_mapping[last_pos : last_pos + 1]
            kv_cache = extract_kv_from_layer(kv_layer, last_slot, 1)

            # Reshape from KV cache format back to per-layer hidden states.
            # Result shape: [num_hidden_states, 1, hidden_size]
            hidden_states = reshape_hidden_states_from_kv_cache(
                kv_cache, self.num_hidden_states
            )
            # Squeeze the single-token dim: [num_hidden_states, hidden_size]
            hidden_states = hidden_states.squeeze(1)

            if self._classifier is None:
                continue

            # Build per-layer dict for the classifier (numpy on CPU).
            layer_ids = getattr(
                self._vllm_config.speculative_config.draft_model_config.hf_config,
                "eagle_aux_hidden_state_layer_ids",
                [],
            )
            hs_np = hidden_states.detach().cpu().float().numpy()
            per_layer: dict[int, Any] = {}
            for i, layer_id in enumerate(layer_ids):
                per_layer[layer_id] = hs_np[i]

            scores = self._classifier.classify(per_layer)
            _shared_scores[request.req_id] = scores
            logger.info(
                "Probe scores for %s: %s",
                request.req_id,
                {k: f"{v:.4f}" for k, v in scores.items()},
            )

    def wait_for_save(self):
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        pass

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = ExampleHiddenStatesConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            meta.add_request(
                new_req.req_id,
                token_ids=token_ids,
                block_ids=new_req.block_ids[0],
                block_size=self._block_size,
            )
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        scores = _shared_scores.pop(req_id, None)

        if scores is not None:
            # Convert int keys to str for msgspec dict[str, Any] compat
            str_scores = {str(k): v for k, v in scores.items()}
            return False, {"probe_scores": str_scores}
        return False, None

    def clear_connector_metadata(self):
        pass

    def real_clear_connector_metadata(self):
        self._connector_metadata = None
