"""Checkpoint loader for adding AtomicPi05 modules to a π0.5 checkpoint."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import download
from openpi.training import weight_loaders


@dataclasses.dataclass(frozen=True)
class AtomicPi05CheckpointLoader(weight_loaders.WeightLoader):
    """Load matching π0.5 tensors and retain initialized atomic-only tensors.

    OpenPI's ordinary ``CheckpointWeightLoader`` only allows absent LoRA
    tensors.  This new policy has deliberately absent modules in a released
    π0.5 checkpoint: Q1--Q4, codebook, latent-control projection and deep
    Action-Expert adapters.  They remain at their prescribed initialization,
    while every matching PaliGemma/Action Expert tensor is restored.
    """

    params_path: str
    trainable_params_path: str | None = None

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        merged = weight_loaders._merge_params(  # noqa: SLF001 - OpenPI's canonical merge routine
            loaded,
            params,
            missing_regex=r".*(queries|codebook|atomic_adapter|coefficient_).*",
        )
        if self.trainable_params_path is not None:
            checkpoint_path = download.maybe_download(self.trainable_params_path)
            with checkpoint_path.open("rb") as handle:
                trainable_params = pickle.load(handle)
            merged = weight_loaders._overwrite_params(merged, trainable_params)  # noqa: SLF001
        return merged
