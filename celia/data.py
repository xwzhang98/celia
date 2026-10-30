"""Training data: feature tables, training views, batches.

``scripts/prepare_data.py`` reads every CAMELS catalog once with ``celia.catalog.read_catalog`` and stores its
feature table (all subhalos above the mass cut) as ``<tables>/<suite>/LH_<id>.npz``. Training only reads these
tables.

One training example is one LH simulation seen twice: the hydrodynamical catalog and its N-body twin (same
initial conditions, same parameters). Each is reduced to a *training view*, a random subset of its subhalos
(``training_rows``). One batch holds 32 simulations of one suite, so 64 catalogs; suites alternate
TNG -> SIMBA -> Astrid from batch to batch.

View seeds cycle with period ``view_cycle`` (64 in the paper): in epoch e every catalog uses view seed
e mod 64, so each catalog has 64 different views. With ``view_cycle=0`` every epoch draws new views.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HYDRO_SUITES = ("IllustrisTNG", "SIMBA", "Astrid")
NBODY_TWIN = {suite: f"{suite}_DM" for suite in HYDRO_SUITES}
BATCH_SIMULATIONS = 32

# A view is drawn like this: pick a reference set of at most 4096 subhalos, then 1024 of them; do this twice and
# keep a random half of the union. Large catalogs therefore give about 1000 rows, small ones about half their
# subhalos.
REFERENCE_SIZE = 4096
DRAW_SIZE = 1024


def split_ids(ids: list[int], seed: int = 0) -> dict[str, list[int]]:
    """Split simulation ids 70 / 10 / 20 into train / val / test (the split of the paper)."""
    order = np.random.default_rng(seed).permutation(sorted(ids))
    n_train, n_val = round(0.7 * len(ids)), round(0.1 * len(ids))
    return {
        "train": sorted(order[:n_train].tolist()),
        "val": sorted(order[n_train : n_train + n_val].tolist()),
        "test": sorted(order[n_train + n_val :].tolist()),
    }


def table_path(tables: Path, suite: str, sim: str) -> Path:
    return Path(tables) / suite / f"{sim}.npz"


ROW_KEYS = ("values", "available", "valid", "host_id")  # one entry per subhalo
CATALOG_KEYS = ("box_size_mpc_h", "particle_grid_side")  # one number per catalog


def load_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {key: z[key] for key in ROW_KEYS + CATALOG_KEYS}


def training_rows(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one training view of a catalog with ``n_rows`` subhalos."""
    if n_rows <= DRAW_SIZE:
        pool = np.arange(n_rows)
    else:
        draws = []
        for _ in range(2):
            reference = rng.choice(n_rows, size=min(n_rows, REFERENCE_SIZE), replace=False)
            draws.append(rng.choice(reference, size=DRAW_SIZE, replace=False))
        pool = np.union1d(*draws)
    return np.sort(rng.choice(pool, size=(len(pool) + 1) // 2, replace=False))


class PairedCatalogs(torch.utils.data.Dataset):
    """Hydro catalogs and their N-body twins, indexed by (suite, LH id)."""

    def __init__(self, tables: Path, train_ids: dict[str, list[int]], view_cycle: int = 64, seed: int = 0):
        self.tables = Path(tables)
        self.records = [(suite, sim_id) for suite in HYDRO_SUITES for sim_id in train_ids[suite]]
        self.view_cycle = view_cycle
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def _view(self, suite: str, sim_id: int, is_nbody: bool) -> dict[str, np.ndarray]:
        table = load_table(table_path(self.tables, suite, f"LH_{sim_id}"))
        view = self.epoch % self.view_cycle if self.view_cycle else self.epoch
        suite_index = HYDRO_SUITES.index(suite.removesuffix("_DM"))
        rng = np.random.default_rng([self.seed, suite_index, sim_id, view, int(is_nbody)])
        rows = training_rows(len(table["values"]), rng)
        return {key: (table[key][rows] if key in ROW_KEYS else table[key]) for key in table}

    def __getitem__(self, index: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        suite, sim_id = self.records[index]
        return self._view(suite, sim_id, False), self._view(NBODY_TWIN[suite], sim_id, True)


class SuiteAlternatingSampler(torch.utils.data.Sampler):
    """Batches of 32 simulations from one suite; suites alternate TNG -> SIMBA -> Astrid.

    Each epoch shuffles the simulations of every suite and drops the incomplete last batch (700 // 32 = 21 batches
    per suite, 63 updates per epoch).
    """

    def __init__(self, dataset: PairedCatalogs, seed: int = 0):
        self.by_suite = {
            suite: [i for i, (s, _) in enumerate(dataset.records) if s == suite] for suite in HYDRO_SUITES
        }
        self.seed = seed
        self.epoch = 0
        self.skip = 0  # batches of the current epoch already trained on (set when a run is continued)

    def __iter__(self):
        per_suite = []
        for k, suite in enumerate(HYDRO_SUITES):
            indices = list(self.by_suite[suite])
            random.Random(self.seed + 1_000_003 * k + self.epoch).shuffle(indices)
            full_batches = range(0, len(indices) - BATCH_SIMULATIONS + 1, BATCH_SIMULATIONS)
            per_suite.append([indices[i : i + BATCH_SIMULATIONS] for i in full_batches])
        interleaved = [batch for batches in zip(*per_suite) for batch in batches]
        yield from interleaved[self.skip :]

    def __len__(self) -> int:
        return len(HYDRO_SUITES) * min(len(v) // BATCH_SIMULATIONS for v in self.by_suite.values())


@dataclass
class Batch:
    """Catalogs padded to a common number of rows. Hydro catalogs come first, then their N-body twins."""

    values: torch.Tensor  # [B, N, F] raw values
    available: torch.Tensor  # [B, N, F]
    valid: torch.Tensor  # [B, N, F]
    padding: torch.Tensor  # [B, N] True on padding rows
    host_id: torch.Tensor  # [B, N] -1 for padding and for subhalos without host
    box_size_mpc_h: torch.Tensor  # [B]
    particle_grid_side: torch.Tensor  # [B] particles per box side of the simulation (CAMELS: 256)

    def to(self, device: torch.device) -> "Batch":
        return Batch(*(getattr(self, f).to(device, non_blocking=True) for f in self.__dataclass_fields__))


def collate(catalogs: list[dict[str, np.ndarray]]) -> Batch:
    """Pad a list of catalogs (dicts from ``load_table``) to one ``Batch``."""
    rows = max(len(c["values"]) for c in catalogs)
    features = catalogs[0]["values"].shape[1]
    values = np.zeros((len(catalogs), rows, features), np.float32)
    available = np.zeros((len(catalogs), rows, features), bool)
    valid = np.zeros((len(catalogs), rows, features), bool)
    padding = np.ones((len(catalogs), rows), bool)
    host_id = np.full((len(catalogs), rows), -1, np.int64)
    for i, c in enumerate(catalogs):
        n = len(c["values"])
        values[i, :n], available[i, :n], valid[i, :n] = c["values"], c["available"], c["valid"]
        padding[i, :n] = False
        host_id[i, :n] = c["host_id"]
    box = np.array([float(c["box_size_mpc_h"]) for c in catalogs], np.float32)
    grid_side = np.array([float(c["particle_grid_side"]) for c in catalogs], np.float32)
    arrays = (values, available, valid, padding, host_id, box, grid_side)
    return Batch(*(torch.from_numpy(a) for a in arrays))


def collate_pairs(pairs: list[tuple[dict, dict]]) -> Batch:
    return collate([hydro for hydro, _ in pairs] + [nbody for _, nbody in pairs])
