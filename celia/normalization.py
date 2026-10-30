"""Per-column normalisation of the subhalo properties.

Each column is standardised, ``(x - mean) / std``, with statistics fitted once on the training catalogs (hydro and
N-body, no parameter labels). Two exceptions:

* positions stay box fractions (mean 0, std 1), because the spatial layers need them in [0, 1);
* each other 3-vector (velocity, host-relative position and velocity) gets mean 0 and one common std, so that a
  random rotation of the catalog does not change the scale of any component.
"""

from __future__ import annotations

import numpy as np
import torch

from celia.features import NUM_COLUMNS, VECTORS


def fit_statistics(tables: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std per column over all available and valid entries of the given feature tables."""
    count = np.zeros(NUM_COLUMNS)
    total = np.zeros(NUM_COLUMNS)
    total_sq = np.zeros(NUM_COLUMNS)
    for table in tables:
        usable = table["available"] & table["valid"]
        x = np.where(usable, table["values"], 0.0).astype(np.float64)
        count += usable.sum(0)
        total += x.sum(0)
        total_sq += (x * x).sum(0)
    n = np.maximum(count, 1)
    mean = total / n
    second_moment = total_sq / n
    std = np.sqrt(np.maximum(second_moment - mean**2, 1e-12))

    for name, cols in VECTORS.items():
        cols = list(cols)
        if name == "position":
            mean[cols], std[cols] = 0.0, 1.0
        else:
            pooled = (count[cols] * second_moment[cols]).sum() / count[cols].sum()
            mean[cols], std[cols] = 0.0, np.sqrt(pooled)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (values - mean) / std.clamp_min(1e-6)
