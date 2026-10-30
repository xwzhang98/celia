"""Small CPU tests of the invariants the method relies on. Run with ``pytest tests``."""

import numpy as np
import pytest
import torch

from celia import dequantize
from celia.data import Batch, collate, split_ids, training_rows
from celia.embed import blocks, covariance_embedding
from celia.features import BARYON_COLUMNS, NUM_COLUMNS, TOKEN_GROUPS
from celia.masking import augment_symmetries, sample_masks
from celia.model import CatalogEncoder, ModelConfig


def random_catalogs(batch: int = 2, rows: int = 40, seed: int = 0) -> Batch:
    rng = np.random.default_rng(seed)
    catalogs = []
    for b in range(batch):
        n = rows - 7 * b
        values = rng.normal(size=(n, NUM_COLUMNS)).astype(np.float32)
        values[:, TOKEN_GROUPS["position"]] = rng.random((n, 3))
        available = np.ones((n, NUM_COLUMNS), bool)
        if b % 2:  # an N-body catalog: no baryons
            available[:, BARYON_COLUMNS] = False
        catalogs.append({
            "values": values, "available": available, "valid": rng.random((n, NUM_COLUMNS)) > 0.05,
            "host_id": rng.integers(-1, 6, n), "box_size_mpc_h": 25.0, "particle_grid_side": 256.0,
        })
    batch_ = collate(catalogs)
    batch_.valid[..., TOKEN_GROUPS["position"]] = ~batch_.padding[..., None]
    return batch_


def small_encoder() -> CatalogEncoder:
    torch.manual_seed(0)
    config = ModelConfig(dim=32, heads=4, inducing_points=8, isab_blocks=2, latent_slots=4, voxel_resolution=4,
                         voxel_channels=8, activation_checkpointing=False)
    return CatalogEncoder(config).eval()


def encode(model, b, observed=None):
    observed = b.available & b.valid if observed is None else observed
    return model(b.values, b.available, b.valid, observed, b.padding, b.box_size_mpc_h, b.particle_grid_side)


def test_split_is_70_10_20():
    split = split_ids(list(range(1000)))
    assert [len(split[k]) for k in ("train", "val", "test")] == [700, 100, 200]
    assert not set(split["train"]) & set(split["test"])


def test_mass_jitter_is_unchanged():
    # Values of the released model's data; any change to the hash, its labels or the draws breaks this.
    subhalo = dequantize.mass_offset_dex(np.array([0, 1, 7]), "IllustrisTNG", "LH_0", dequantize.SUBHALO_LABEL)
    host = dequantize.mass_offset_dex(np.array([0, 3]), "SIMBA", "LH_5", dequantize.HOST_LABEL)
    np.testing.assert_array_equal(subhalo, np.array([0.03939899802207947, -0.014302928000688553,
                                                      -0.011809129267930984], np.float32))
    np.testing.assert_array_equal(host, np.array([-0.02019660174846649, 0.004964449442923069], np.float32))


@pytest.mark.parametrize("n", [10, 1024, 1500, 9000])
def test_training_view_is_about_half_of_two_draws(n):
    rows = training_rows(n, np.random.default_rng(0))
    assert len(set(rows)) == len(rows) and rows.max() < n
    assert len(rows) <= 1024 and len(rows) >= min(n, 1024) // 2


def test_blocks_cover_every_subhalo_once():
    for n in (5, 1024, 3000):
        scored = np.concatenate([s for s, _ in blocks(n, catalog_number=7)])
        assert sorted(scored) == list(range(n))
        assert all(len(rows) == min(n, 1024) for _, rows in blocks(n, 7))


def test_covariance_embedding_is_an_isometry():
    a, b = torch.randn(50, 6), torch.randn(50, 6)
    cov = lambda x: (x - x.mean(0)).T @ (x - x.mean(0)) / len(x)  # noqa: E731
    distance = torch.linalg.norm(covariance_embedding(a) - covariance_embedding(b))
    assert torch.allclose(distance, torch.linalg.norm(cov(a) - cov(b)), rtol=1e-5)


def test_encoder_is_permutation_equivariant_and_ignores_padding():
    model, b = small_encoder(), random_catalogs()
    out = encode(model, b)
    perm = torch.randperm(b.values.shape[1])
    shuffled = type(b)(*(getattr(b, f)[:, perm] if getattr(b, f).ndim > 1 else getattr(b, f)
                         for f in b.__dataclass_fields__))
    out_perm = encode(model, shuffled)
    assert torch.allclose(out.object_features[:, perm], out_perm.object_features, atol=1e-5)
    assert torch.allclose(out.latent_slots, out_perm.latent_slots, atol=1e-5)
    assert (out.object_features[b.padding] == 0).all()


def test_hidden_values_never_reach_the_model():
    model, b = small_encoder(), random_catalogs()
    observed = b.available & b.valid
    observed[..., TOKEN_GROUPS["velocity"]] = False
    garbage = b.values.clone()
    garbage[..., TOKEN_GROUPS["velocity"]] = float("nan")
    clean = encode(model, b, observed).object_features
    dirty = model(garbage, b.available, b.valid, observed, b.padding, b.box_size_mpc_h,
                  b.particle_grid_side).object_features
    assert torch.equal(clean, dirty)


def test_masks_respect_the_rules():
    b = random_catalogs(batch=4, rows=60)
    generator = torch.Generator().manual_seed(1)
    values, available, valid = augment_symmetries(b.values, b.available, b.valid, b.padding, generator)
    view = sample_masks(values, available, valid, b.padding, b.host_id, torch.Generator().manual_seed(2))
    usable = available & valid & ~b.padding[..., None]
    assert not (view.observed & ~usable).any()  # nothing appears that was not there
    assert not (view.targets & view.observed).any()  # targets are hidden
    assert not view.targets[..., TOKEN_GROUPS["position"]].any()
    positions = values[..., TOKEN_GROUPS["position"]][~b.padding]
    assert (positions >= 0).all() and (positions < 1).all()
