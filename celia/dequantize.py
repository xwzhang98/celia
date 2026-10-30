"""Fixed log-mass jitter that hides the particle-mass lattice.

A subhalo or FoF group mass in an N-body or hydro run is a whole number of dark-matter particles times the particle
mass, and the particle mass is proportional to Omega_m - Omega_b. The exact lattice of allowed masses therefore
reveals Omega_m without any physics. To remove this shortcut, every mass column gets a small fixed offset in dex,
drawn from a Gaussian with standard deviation 0.02 dex and clipped at +-0.05 dex.

The offset is a deterministic function of (suite, simulation, coordinate, object id): it is the same in every
training view and every epoch, so it is a fixed dequantisation, not an augmentation. Subhalo masses are keyed by the
HDF5 subhalo row and the column name (independent draws per column). The three host masses (FoF, M200c, M500c) are
keyed by the host group id and the shared label "host", so they move together and stay mutually consistent.

The hash, the random-number mixer and the float32 rounding below must not change: the released embeddings depend on
these exact numbers.
"""

from __future__ import annotations

import hashlib

import numpy as np

CONTRACT = "physical_mass_lattice_jitter_v3_host_shared"
SEED = 220906843
STD_DEX = np.float32(0.02)
CLIP_DEX = np.float32(0.05)
# Labels that enter the hash. They are fixed strings (the subhalo label is an old column name), so changing
# them would change every jitter draw.
SUBHALO_LABEL = "log10_dm_mass_1e10msunh"
HOST_LABEL = "host"


def stable_uint64(*parts: object) -> int:
    """A 63-bit integer hash of the parts (sha256 of their text joined by the unit separator)."""
    text = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """The splitmix64 finaliser: a fast, well-mixed integer hash (wraps modulo 2**64)."""
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def standard_normal(keys: np.ndarray, suite: str, sim: str, coordinate: str) -> np.ndarray:
    """One N(0, 1) float32 draw per non-negative integer key, via two hashed uniforms and Box-Muller."""
    keys = np.asarray(keys, dtype=np.int64)
    if (keys < 0).any():
        raise ValueError("jitter keys must be non-negative")
    base = np.uint64(stable_uint64(CONTRACT, SEED, suite, sim, coordinate))
    mixed = keys.astype(np.uint64) ^ base
    first = _splitmix64(mixed)
    second = _splitmix64(mixed ^ np.uint64(0xD1B54A32D192ED03))
    # The top 53 bits give a uniform number in (0, 1) that is exact in float64.
    u1 = ((first >> np.uint64(11)).astype(np.float64) + 0.5) / float(2**53)
    u2 = ((second >> np.uint64(11)).astype(np.float64) + 0.5) / float(2**53)
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)).astype(np.float32)


def mass_offset_dex(keys: np.ndarray, suite: str, sim: str, coordinate: str) -> np.ndarray:
    """The float32 log10-mass offset of each key: 0.02 dex times N(0, 1), clipped at +-0.05 dex."""
    return np.clip(STD_DEX * standard_normal(keys, suite, sim, coordinate), -CLIP_DEX, CLIP_DEX)
