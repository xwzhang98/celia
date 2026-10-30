"""Write the Celia embeddings of a set of CAMELS catalogs.

    python scripts/export_embeddings.py --weights celia_step18000.pt --camels /path/to/CAMELS \
        --suite IllustrisTNG --set LH --out embeddings/

Writes ``<out>/<suite>_<set>.npy`` (float32, one 32,896-d row per catalog, in ascending simulation index) and
``<out>/<suite>_<set>_names.txt`` (the catalog of each row). Works for any suite in the CAMELS Subfind format,
including Swift-EAGLE and the N-body suites (``--suite IllustrisTNG_DM``), and for the LH, CV and 1P sets.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from celia.catalog import read_catalog
from celia.embed import load_encoder, embed


def catalog_number(name: str) -> int:
    """The index that seeds the block partition: i for LH_i / CV_i, the running number for 1P names."""
    match = re.fullmatch(r"(?:LH|CV)_(\d+)", name)
    return int(match.group(1)) if match else -1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--camels", type=Path, required=True, help="CAMELS root directory")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--set", default="LH", choices=("LH", "CV", "1P"))
    parser.add_argument("--out", type=Path, default=Path("embeddings"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = load_encoder(args.weights, args.device)
    root = args.camels / "FOF_Subfind" / args.suite / "L25n256" / args.set
    natural_order = lambda name: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]  # noqa: E731
    names = sorted((p.parent.name for p in root.glob("*/groups_090.hdf5")), key=natural_order)
    vectors = []
    for k, name in enumerate(names):
        catalog = read_catalog(root / name / "groups_090.hdf5", args.suite, name, nbody=args.suite.endswith("_DM"))
        number = catalog_number(name)
        vectors.append(embed(model, catalog, number if number >= 0 else k))
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / f"{args.suite}_{args.set}.npy", np.stack(vectors))
    (args.out / f"{args.suite}_{args.set}_names.txt").write_text("\n".join(names) + "\n")
    print(f"{len(names)} embeddings -> {args.out}")


if __name__ == "__main__":
    main()
