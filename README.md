# Celia: Catalog Embeddings Learned from cosmologIcal simulAtions

Code for the paper [*Celia: Catalog Embeddings Learned from cosmologIcal simulAtions*](https://arxiv.org/abs/2610.40218)
(Zhang, Ho, Kumar, Di Matteo, Croft, Póczos 2026).

Celia is a self-supervised encoder that maps a catalog of subhalos and their galaxy properties to one embedding
vector. It is pretrained without parameter labels on the IllustrisTNG, SIMBA and Astrid suites of
[CAMELS](https://camels.readthedocs.io) and their N-body twins, by reconstructing masked subhalo properties and
predicting the representations of an EMA teacher.

This repository contains the two parts needed to reproduce the model and its embeddings: **pretraining** and the
**embedding export**. The analyses of the paper (linear read-outs, figures) work on the exported embeddings.

## Install

```bash
pip install -e .            # Python >= 3.11, PyTorch >= 2.8
pytest tests                # CPU checks
```

## Use the trained model

Download `celia_step18000.pt` from the [release page](../../releases) (student encoder and normalisation
statistics). Then, for any catalog in the CAMELS Subfind format:

```python
from celia.catalog import read_catalog
from celia.embed import embed, load_encoder

model = load_encoder("celia_step18000.pt", device="cuda")
catalog = read_catalog("CAMELS/FOF_Subfind/IllustrisTNG/L25n256/LH/LH_0/groups_090.hdf5",
                       suite="IllustrisTNG", sim="LH_0", nbody=False)
vector = embed(model, catalog, catalog_number=0)      # numpy float32, shape (32896,)
```

or for a whole set:

```bash
python scripts/export_embeddings.py --weights celia_step18000.pt --camels /path/to/CAMELS \
    --suite IllustrisTNG --set LH --out embeddings/
```

`suite` and `sim` only seed the fixed mass dequantisation (see `celia/dequantize.py`); `catalog_number` only seeds
the random split of a large catalog into blocks of 1024. Both make the embedding reproducible.

## Reproduce the pretraining

1. Download the z = 0 group catalogs (`groups_090.hdf5`) of the LH sets of IllustrisTNG, SIMBA, Astrid and of
   their N-body twins IllustrisTNG_DM, SIMBA_DM, Astrid_DM (6 x 1000 files) from CAMELS.
2. Build the feature tables (CPU):
   ```bash
   python scripts/prepare_data.py --camels /path/to/CAMELS --out data/tables --workers 16
   ```
3. Initialise a run (fits the normalisation statistics on the training simulations):
   ```bash
   python scripts/init_model.py --config configs/celia.yaml --run runs/celia
   ```
4. Train on one GPU:
   ```bash
   python scripts/train.py --run runs/celia --stop 10000    # warmup and constant learning rate
   python scripts/train.py --run runs/celia --stop 18000    # linear decay to zero
   ```
   The second command continues from the first. `runs/celia/step18000.pt` is the final model;
   `runs/celia/log.jsonl` has one line per update.

A new run starts from a new random initialisation, so it reproduces the paper model statistically, not bit for
bit (see `docs/VERIFICATION.md`).

## Code map

| file | what it does |
|---|---|
| `celia/features.py` | the 53 subhalo properties and their 22 token groups |
| `celia/catalog.py`, `celia/dequantize.py` | CAMELS HDF5 file -> feature table (all subhalos above the mass cut) |
| `celia/normalization.py` | per-column standardisation |
| `celia/model.py` | the encoder and its two training heads |
| `celia/masking.py` | cube symmetries and the masking rules |
| `celia/losses.py` | masked-property loss M and teacher-prediction loss O |
| `celia/data.py` | training views, batches of hydro catalogs and N-body twins |
| `celia/train.py` | optimizer, schedules, EMA teacher, checkpoints |
| `celia/embed.py` | the 32,896-d covariance embedding of a catalog |

## Citation

```bibtex
@article{zhang2026celia,
  title   = {Celia: Catalog Embeddings Learned from cosmologIcal simulAtions},
  author  = {Zhang, Xiaowen and Ho, Matthew and Kumar, Anshul and Di Matteo, Tiziana and Croft, Rupert A. C.
             and P{\'o}czos, Barnab{\'a}s},
  journal = {arXiv e-prints},
  eprint  = {2610.40218},
  archivePrefix = {arXiv},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`).
