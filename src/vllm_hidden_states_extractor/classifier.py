import os
import pickle

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


class HiddenStateClassifier:
    """Runs pre-trained per-layer sklearn probes on hidden state vectors.

    Each probe is a pipeline of PCA → StandardScaler → LogisticRegression
    that produces a refuse probability for a single layer's hidden state.

    This module is vLLM-agnostic — it operates on numpy arrays only.
    """

    def __init__(self, pkl_path: str):
        logger.info("Loading layer probes from %s", pkl_path)
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)

        self.probes: dict = bundle["probes"]
        self.probe_layers: list[int] = bundle["probe_layers"]
        logger.info(
            "Loaded %d probes for layers %s", len(self.probes), self.probe_layers
        )

    def classify(self, hidden_states: dict[int, np.ndarray]) -> dict[int, float]:
        """Run probes on per-layer hidden states.

        Args:
            hidden_states: layer_idx → numpy array of shape [hidden_size].

        Returns:
            layer_idx → refuse probability (float).
        """
        scores: dict[int, float] = {}
        for layer_idx in sorted(self.probes.keys()):
            if layer_idx not in hidden_states:
                logger.warning("Missing hidden state for layer %d, defaulting to 0.5", layer_idx)
                scores[layer_idx] = 0.5
                continue

            probe = self.probes[layer_idx]
            h = hidden_states[layer_idx].reshape(1, -1)

            h_pca = probe["pca"].transform(h)
            h_scaled = probe["scaler"].transform(h_pca)
            refuse_prob = float(probe["lr"].predict_proba(h_scaled)[0, 1])
            scores[layer_idx] = refuse_prob

        return scores

    @staticmethod
    def discover(classifier_dir: str, model_name: str) -> "HiddenStateClassifier | None":
        """Look up a classifier for the given model.

        Args:
            classifier_dir: Root directory containing per-model subdirectories.
            model_name: HuggingFace model name (e.g. "meta-llama/Llama-3.1-8B-Instruct").

        Returns:
            A HiddenStateClassifier if found, else None.
        """
        sanitized = model_name.replace("/", "--")
        pkl_path = os.path.join(classifier_dir, sanitized, "layer_probes.pkl")
        if not os.path.isfile(pkl_path):
            logger.warning(
                "No classifier found for model %s (looked at %s). "
                "Classification will be skipped.",
                model_name,
                pkl_path,
            )
            return None
        return HiddenStateClassifier(pkl_path)
