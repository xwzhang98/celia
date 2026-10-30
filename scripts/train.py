"""Train a run up to a given update (one GPU).

    python scripts/train.py --run runs/celia_seed0 --stop 10000     # main phase (constant learning rate)
    python scripts/train.py --run runs/celia_seed0 --stop 18000     # decay phase

The run continues from its newest checkpoint, so the second command picks up where the first one stopped. The
paper model is ``step18000.pt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from celia.train import load_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory made by scripts/init_model.py")
    parser.add_argument("--stop", type=int, required=True, help="train until this update")
    args = parser.parse_args()
    train(args.run, load_config(args.run / "config.yaml"), args.stop)


if __name__ == "__main__":
    main()
