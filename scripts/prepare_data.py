"""Read the CAMELS LH catalogs once and store their feature tables for training.

    python scripts/prepare_data.py --camels /path/to/CAMELS --out data/tables

Expects the CAMELS layout ``<camels>/FOF_Subfind/<suite>/L25n256/LH/LH_<i>/groups_090.hdf5`` (z = 0) for the
suites IllustrisTNG, SIMBA, Astrid and their N-body twins IllustrisTNG_DM, SIMBA_DM, Astrid_DM; another layout can
be given with ``--pattern``. Writes one
``<out>/<suite>/LH_<i>.npz`` per catalog; existing files are skipped, so the script can be rerun after an
interruption.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from celia.catalog import read_catalog
from celia.data import HYDRO_SUITES, NBODY_TWIN

PARTICLE_GRID_SIDE = 256  # CAMELS L25n256


def write_table(source: Path, out: Path, suite: str, sim: str) -> None:
    if out.exists():
        return
    catalog = read_catalog(source, suite, sim, nbody=suite.endswith("_DM"))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        values=catalog.values,
        available=catalog.available,
        valid=catalog.valid,
        host_id=catalog.host_id,
        box_size_mpc_h=np.float32(catalog.box_size_mpc_h),
        particle_grid_side=np.float32(PARTICLE_GRID_SIDE),
    )
    tmp.replace(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camels", type=Path, required=True, help="CAMELS root directory")
    parser.add_argument("--pattern", default="{camels}/FOF_Subfind/{suite}/L25n256/LH/{sim}/groups_090.hdf5")
    parser.add_argument("--out", type=Path, default=Path("data/tables"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    jobs = []
    for suite in HYDRO_SUITES + tuple(NBODY_TWIN.values()):
        for i in range(1000):
            sim = f"LH_{i}"
            source = Path(args.pattern.format(camels=args.camels, suite=suite, sim=sim))
            jobs.append((source, args.out / suite / f"{sim}.npz", suite, sim))
    with ProcessPoolExecutor(args.workers) as pool:
        for future in [pool.submit(write_table, *job) for job in jobs]:
            future.result()
    print(f"{len(jobs)} tables in {args.out}")


if __name__ == "__main__":
    main()
