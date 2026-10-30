"""The Celia embedding of a catalog.

Every subhalo above the mass cut is encoded, with nothing masked. Catalogs are larger than the encoder's training
context, so the subhalos are put in a random order and cut into blocks of 1024; each block is encoded on its own,
and the per-subhalo features C are put back in place. A short last block is filled up to 1024 with subhalos from
the start of the order; their outputs are discarded, so every subhalo contributes exactly once.

The embedding is the covariance of C over all subhalos of the catalog (256 x 256, divided by N), written as its
upper triangle with off-diagonal entries multiplied by sqrt(2): 256 * 257 / 2 = 32,896 numbers. With this scaling
the Euclidean distance between two embeddings equals the Frobenius distance between the two covariance matrices.

Usage::

    model = load_encoder("celia_step18000.pt", device="cuda")
    vector = embed(model, read_catalog(path, "IllustrisTNG", "LH_0", nbody=False), catalog_number=0)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from celia.catalog import Catalog
from celia.model import CatalogEncoder, ModelConfig
from celia.normalization import normalize

BLOCK_SIZE = 1024
BLOCKS_PER_FORWARD = 4
EMBEDDING_DIM = 256 * 257 // 2


@dataclass
class TrainedEncoder:
    encoder: CatalogEncoder
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    device: torch.device


def load_encoder(checkpoint: str | Path, device: str = "cuda") -> TrainedEncoder:
    """The student encoder and its normalisation statistics from a released or training checkpoint."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = CatalogEncoder(ModelConfig(**state["config"]["model"]) if "config" in state else ModelConfig())
    encoder.load_state_dict(state["encoder"])
    encoder = encoder.to(device).eval().requires_grad_(False)
    mean, std = state["feature_mean"].to(device), state["feature_std"].to(device)
    return TrainedEncoder(encoder, mean, std, torch.device(device))


def blocks(n_subhalos: int, catalog_number: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(scored rows, encoded rows) of each block: a fixed random partition of the catalog into blocks of 1024.

    ``catalog_number`` (the LH, CV or 1P index) only seeds the random order, so the result is reproducible.
    """
    seed = (20_260_729 + 1_000_003 * catalog_number + 9_176 * n_subhalos) % 2**32
    order = np.random.default_rng(seed).permutation(n_subhalos)
    result = []
    for start in range(0, n_subhalos, BLOCK_SIZE):
        scored = order[start : start + BLOCK_SIZE]
        fill = order[: min(BLOCK_SIZE - len(scored), n_subhalos - len(scored))]
        result.append((scored, np.concatenate([scored, fill])))
    return result


def covariance_embedding(features: torch.Tensor) -> torch.Tensor:
    """Upper triangle of the covariance of ``features`` [N, D] with sqrt(2) off-diagonals, [D (D + 1) / 2]."""
    centered = features - features.mean(0)
    covariance = centered.T @ centered / features.shape[0]
    i, j = torch.triu_indices(features.shape[1], features.shape[1], device=features.device)
    upper = covariance[i, j]
    return torch.where(i == j, upper, upper * 2**0.5)


@torch.no_grad()
def object_features(
    model: TrainedEncoder, catalog: Catalog, catalog_number: int, particle_grid_side: int = 256
) -> torch.Tensor:
    """Per-subhalo features C [N, 256] of a whole catalog, encoded block by block (float32).

    ``particle_grid_side`` is the number of particles per box side of the simulation (CAMELS L25n256: 256).
    """
    device = model.device
    values = normalize(torch.from_numpy(catalog.values).to(device), model.feature_mean, model.feature_std)
    available = torch.from_numpy(catalog.available).to(device)
    valid = torch.from_numpy(catalog.valid).to(device)
    n = len(values)
    features = torch.zeros(n, model.encoder.config.dim, device=device)
    plan = blocks(n, catalog_number)
    for first in range(0, len(plan), BLOCKS_PER_FORWARD):
        group = plan[first : first + BLOCKS_PER_FORWARD]
        width = max(len(rows) for _, rows in group)
        x = torch.zeros(len(group), width, values.shape[1], device=device)
        avail = torch.zeros(len(group), width, values.shape[1], dtype=torch.bool, device=device)
        val = torch.zeros_like(avail)
        padding = torch.ones(len(group), width, dtype=torch.bool, device=device)
        for k, (_, rows) in enumerate(group):
            rows_t = torch.from_numpy(rows).to(device)
            x[k, : len(rows)] = values[rows_t]
            avail[k, : len(rows)] = available[rows_t]
            val[k, : len(rows)] = valid[rows_t]
            padding[k, : len(rows)] = False
        box = torch.full((len(group),), catalog.box_size_mpc_h, device=device)
        grid_side = torch.full((len(group),), float(particle_grid_side), device=device)
        out = model.encoder(x, avail, val, avail & val, padding, box, grid_side).object_features
        for k, (scored, _) in enumerate(group):
            features[torch.from_numpy(scored).to(device)] = out[k, : len(scored)]
    return features


def embed(
    model: TrainedEncoder, catalog: Catalog, catalog_number: int, particle_grid_side: int = 256
) -> np.ndarray:
    """The 32,896-d Celia embedding of one catalog (float32)."""
    features = object_features(model, catalog, catalog_number, particle_grid_side)
    return covariance_embedding(features).cpu().numpy()
