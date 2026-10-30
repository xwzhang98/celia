"""Create a new run: a freshly initialised model and the normalisation statistics.

    python scripts/init_model.py --config configs/celia.yaml --run runs/celia_seed0

Writes ``<run>/config.yaml`` and ``<run>/step0.pt``. The weights are PyTorch's default initialisation drawn with
the config's seed; the statistics are fitted on the training catalogs of the three hydro suites and their N-body
twins (no parameter labels are read).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from celia.data import HYDRO_SUITES, NBODY_TWIN, load_table, split_ids, table_path
from celia.normalization import fit_statistics
from celia.train import Celia, load_config, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="new run directory (must not exist)")
    args = parser.parse_args()

    config = load_config(args.config)
    args.run.mkdir(parents=True)
    shutil.copy(args.config, args.run / "config.yaml")

    train_ids = split_ids(list(range(1000)), config.split_seed)["train"]
    tables = [
        load_table(table_path(Path(config.tables), suite, f"LH_{i}"))
        for suite in HYDRO_SUITES + tuple(NBODY_TWIN.values())
        for i in train_ids
    ]
    mean, std = fit_statistics(tables)

    torch.manual_seed(config.seed)
    model = Celia(config.model)
    save(args.run / "step0.pt", model, None, 0, torch.from_numpy(mean), torch.from_numpy(std), config)
    print(f"initialised {args.run}")


if __name__ == "__main__":
    main()
