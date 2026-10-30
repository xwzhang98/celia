"""Read one CAMELS group catalog (``groups_090.hdf5``) into a table of 53 numbers per subhalo.

Every subhalo with a dark-matter mass of at least 0.646190315e10 Msun/h
(``SubhaloMassType[:, 1]``) becomes one row, in ascending HDF5 subhalo order. The columns are those of
``celia.features.COLUMNS``; each column also has an ``available`` flag (the run provides the field) and a ``valid``
flag (the value is physically defined). Values are 0 wherever ``available & valid`` is False.

The recipe reproduces the original pipeline to float32 rounding: the intermediate float32 values that the old code
stored in its cache are formed here in the same precision before the final float64 transforms. Masses carry the
fixed jitter of ``celia.dequantize``. Host-relative positions, velocities and the host centre offset are expressed
in units of the host R200c and V200c implied by the *jittered* M200c, so they reveal no more than the mass does.

N-body runs (``nbody=True``) have no baryons: the baryonic columns are unavailable, and the host mass column holds
the FoF dark-matter mass instead of the total FoF mass (the two are equal in an N-body run).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from celia import dequantize
from celia.features import COLUMN_INDEX, NUM_COLUMNS

# Completeness cut: 50 times the largest DM particle mass of the LH runs, in 1e10 Msun/h. A physical mass cut
# (not a particle-count cut) keeps the selected population independent of the particle mass, hence of Omega_m.
MIN_DM_MASS_1E10MSUNH = 0.646190315

# Spherical-overdensity constants at z = 0, in Msun/h, comoving kpc/h and km/s (h cancels in V200c).
CRITICAL_DENSITY_MSUNH_PER_CKPCH3 = 277.536627
GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN = 4.30091e-6
OVERDENSITY = 200.0


@dataclass(frozen=True)
class Catalog:
    """All eligible subhalos of one catalog. Arrays have one row per subhalo, in ascending ``row_index``."""

    suite: str
    sim: str
    nbody: bool
    box_size_mpc_h: float
    row_index: np.ndarray  # int64 [n], subhalo row in the HDF5 file
    host_id: np.ndarray  # int64 [n], SubhaloGrNr, -1 if the host link is invalid
    values: np.ndarray  # float32 [n, 53]
    available: np.ndarray  # bool [n, 53]
    valid: np.ndarray  # bool [n, 53]


class _Table:
    """The 53 columns under construction. A column is stored as 0 wherever it is not available and valid."""

    def __init__(self, n: int) -> None:
        self.values = np.zeros((n, NUM_COLUMNS), dtype=np.float32)
        self.available = np.zeros((n, NUM_COLUMNS), dtype=bool)
        self.valid = np.zeros((n, NUM_COLUMNS), dtype=bool)

    def set(self, name: str, value: np.ndarray, available: np.ndarray, valid: np.ndarray) -> None:
        column = COLUMN_INDEX[name]
        usable = available & valid
        self.values[:, column] = np.where(usable, value, 0.0).astype(np.float32)
        self.available[:, column] = available
        self.valid[:, column] = usable

    def set_log_pair(self, name: str, bit_name: str | None, raw: np.ndarray, available: np.ndarray,
                     defined: np.ndarray, log_offset: np.ndarray | float = 0.0) -> None:
        """A log10 value that is valid only for x > 0, plus a 0/1 "is positive" bit that is valid wherever x is
        defined. A measured zero (no stars, no star formation) is information, distinct from a missing field."""
        positive = defined & (raw > 0.0)
        log_value = np.log10(np.where(positive, raw, 1.0).astype(np.float64)) - log_offset
        self.set(name, log_value, available, positive)
        if bit_name is not None:
            self.set(bit_name, positive.astype(np.float32), available, defined)


def _log10_float32(raw: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """log10 in float32 of a positive quantity, and where it is defined (finite and > 0).

    NumPy's float32 log10 depends on the CPU: with AVX512 dispatch it can differ from glibc log10f by a few ulps.
    The original cache was mostly built without AVX512 (bit-identical to glibc log10f); to reproduce it bit for bit
    on an AVX512 machine, start Python with NPY_DISABLE_CPU_FEATURES set to the AVX512 feature names.
    """
    raw = np.asarray(raw, dtype=np.float32)
    positive = np.isfinite(raw) & (raw > 0.0)
    return np.log10(np.maximum(np.where(positive, raw, 1.0), np.float32(floor))), positive


def _subhalo_field(subhalos: h5py.Group, name: str, rows: np.ndarray, component: int | None = None
                   ) -> tuple[np.ndarray, bool]:
    """float32 values of one Subhalo field at the selected rows, and whether the file has the field."""
    if name not in subhalos:
        return np.zeros(len(rows), dtype=np.float32), False
    data = np.asarray(subhalos[name][()], dtype=np.float32)
    data = data[rows] if component is None else data[rows, component]
    return data, True


def _host_field(groups: h5py.Group, name: str, host_id: np.ndarray, host_ok: np.ndarray,
                component: int | slice | None = None) -> tuple[np.ndarray, np.ndarray]:
    """float32 values of one Group field at each subhalo's host (0 for an invalid host), and its availability."""
    data = np.asarray(groups[name][()], dtype=np.float32)
    if component is not None:
        data = data[:, component]
    values = np.zeros((len(host_id),) + data.shape[1:], dtype=np.float32)
    values[host_ok] = data[host_id[host_ok]]
    return values, host_ok.copy()


def _periodic_distance(a: np.ndarray, b: np.ndarray, box_size: float) -> np.ndarray:
    """Minimum-image distance |a - b| in a periodic box (float64)."""
    delta = a.astype(np.float64) - b.astype(np.float64)
    delta -= np.round(delta / box_size) * box_size
    return np.linalg.norm(delta, axis=1)


def _log1p_ratio_float32(numerator: np.ndarray, denominator: np.ndarray, defined: np.ndarray) -> np.ndarray:
    """float32 log1p(numerator / denominator), the stored form of the two centre-offset columns."""
    ratio = np.zeros(len(numerator), dtype=np.float32)
    np.divide(numerator, denominator, out=ratio, where=defined)
    return np.where(defined, np.log1p(ratio), 0.0).astype(np.float32)


def _host_scales(log10_m200c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R200c [ckpc/h] and V200c [km/s] implied by log10 M200c [1e10 Msun/h]."""
    mass = np.power(10.0, log10_m200c.astype(np.float64)) * 1.0e10
    radius = np.cbrt(3.0 * mass / (4.0 * math.pi * OVERDENSITY * CRITICAL_DENSITY_MSUNH_PER_CKPCH3))
    return radius, np.sqrt(GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN * mass / radius)


def read_catalog(path: str | Path, suite: str, sim: str, nbody: bool) -> Catalog:
    """Read all eligible subhalos of one catalog.

    ``suite`` and ``sim`` (for example "IllustrisTNG", "LH_0") key the mass jitter only; the data come from
    ``path``.
    """
    with h5py.File(path, "r") as file:
        box_size = float(file["Header"].attrs["BoxSize"])  # ckpc/h
        subhalos, groups = file["Subhalo"], file["Group"]

        dm_mass_all = np.asarray(subhalos["SubhaloMassType"][:, 1], dtype=np.float64)
        rows = np.flatnonzero(np.isfinite(dm_mass_all) & (dm_mass_all >= MIN_DM_MASS_1E10MSUNH)).astype(np.int64)
        n = len(rows)
        table = _Table(n)
        everywhere = np.ones(n, dtype=bool)

        host_id = np.asarray(subhalos["SubhaloGrNr"][()], dtype=np.int64)[rows]
        first_subhalo = np.asarray(groups["GroupFirstSub"][()], dtype=np.int64)
        host_ok = (host_id >= 0) & (host_id < len(first_subhalo))
        host_id = np.where(host_ok, host_id, -1)

        # Position (fraction of the box) and peculiar velocity (1000 km/s).
        raw_pos, _ = _subhalo_field(subhalos, "SubhaloPos", rows)
        pos_ok = np.isfinite(raw_pos)
        position = np.mod(raw_pos / np.float32(box_size), np.float32(1.0))
        raw_vel, _ = _subhalo_field(subhalos, "SubhaloVel", rows)
        vel_ok = np.isfinite(raw_vel)
        velocity = raw_vel / np.float32(1000.0)
        for k, axis in enumerate("xyz"):
            table.set(f"position_{axis}_box", position[:, k], everywhere, pos_ok[:, k])
            table.set(f"velocity_{axis}_1000_km_per_s", velocity[:, k], everywhere, vel_ok[:, k])

        # Position and velocity relative to the host group (minimum image), in float32 as first stored.
        host_pos_raw, _ = _host_field(groups, "GroupPos", host_id, host_ok, slice(0, 3))
        host_vel_raw, _ = _host_field(groups, "GroupVel", host_id, host_ok, slice(0, 3))
        relative_pos = position - np.mod(host_pos_raw / np.float32(box_size), np.float32(1.0))
        relative_pos -= np.round(relative_pos)
        relative_vel = velocity - host_vel_raw / np.float32(1000.0)
        relative_pos_ok = pos_ok & np.isfinite(host_pos_raw) & host_ok[:, None]
        relative_vel_ok = vel_ok & np.isfinite(host_vel_raw) & host_ok[:, None]

        is_central_known = host_ok & (first_subhalo[np.maximum(host_id, 0)] >= 0)
        is_central = host_ok & (rows == first_subhalo[np.maximum(host_id, 0)])
        table.set("is_central", is_central.astype(np.float32), is_central_known, is_central_known)

        # Dark-matter mass with its per-subhalo jitter, radii and internal dynamics.
        dm_mass, dm_mass_ok = _log10_float32(_subhalo_field(subhalos, "SubhaloMassType", rows, 1)[0], 1.0e-12)
        dm_mass = dm_mass + np.where(
            dm_mass_ok, dequantize.mass_offset_dex(rows, suite, sim, dequantize.SUBHALO_LABEL), np.float32(0.0))
        table.set("log10_dm_mass_1e10_msun_per_h", dm_mass, everywhere, dm_mass_ok)
        for name, field, component, floor in (
            ("log10_dm_halfmass_radius_ckpc_per_h", "SubhaloHalfmassRadType", 1, 1.0e-6),
            ("log10_subhalo_total_halfmass_radius_ckpc_per_h", "SubhaloHalfmassRad", None, 1.0e-6),
            ("log10_subhalo_vmax_km_per_s", "SubhaloVmax", None, 1.0e-6),
            ("log10_subhalo_velocity_dispersion_km_per_s", "SubhaloVelDisp", None, 1.0e-6),
            ("log10_subhalo_vmax_radius_ckpc_per_h", "SubhaloVmaxRad", None, 1.0e-6),
        ):
            raw, present = _subhalo_field(subhalos, field, rows, component)
            value, ok = _log10_float32(raw, floor)
            table.set(name, value, everywhere & present, ok)
        spin, _ = _subhalo_field(subhalos, "SubhaloSpin", rows)
        spin_norm, spin_ok = _log10_float32(np.linalg.norm(spin, axis=1), 1.0e-12)
        table.set("log10_subhalo_spin_norm", spin_norm, everywhere, spin_ok)

        # Offset between the potential minimum and the centre of mass, in units of the half-mass radius: a
        # relaxation indicator.
        raw_cm, _ = _subhalo_field(subhalos, "SubhaloCM", rows)
        half_radius, _ = _subhalo_field(subhalos, "SubhaloHalfmassRad", rows)
        offset_ok = (pos_ok.all(axis=1) & np.isfinite(raw_cm).all(axis=1)
                     & np.isfinite(half_radius) & (half_radius > 0.0))
        offset = _log1p_ratio_float32(_periodic_distance(raw_pos, raw_cm, box_size), half_radius, offset_ok)
        table.set("log1p_periodic_pos_cm_offset_over_total_halfmass_radius", offset, everywhere, offset_ok)

        # Host masses. One jitter offset per host is shared by its FoF mass, M200c and M500c.
        host_jitter = dequantize.mass_offset_dex(np.maximum(host_id, 0), suite, sim, dequantize.HOST_LABEL)
        host_mass = {}
        for name, field, component in (
            ("log10_host_m200c_1e10_msun_per_h", "Group_M_Crit200", None),
            ("log10_host_m500c_1e10_msun_per_h", "Group_M_Crit500", None),
            ("log10_host_dm_mass_1e10_msun_per_h", "GroupMassType", 1),
        ):
            raw, _ = _host_field(groups, field, host_id, host_ok, component)
            value, ok = _log10_float32(raw, 1.0e-12)
            ok &= host_ok
            host_mass[name] = (value + np.where(ok, host_jitter, np.float32(0.0)), ok)
        for name in ("log10_host_m200c_1e10_msun_per_h", "log10_host_m500c_1e10_msun_per_h"):
            table.set(name, host_mass[name][0], host_ok, host_mass[name][1])

        # Host-relative coordinates in units of R200c and V200c of the jittered M200c.
        log_m200c, m200c_ok = host_mass["log10_host_m200c_1e10_msun_per_h"]
        r200c, v200c = _host_scales(np.where(m200c_ok, log_m200c, np.float32(0.0)))
        for k, axis in enumerate("xyz"):
            table.set(f"host_relative_position_{axis}_r200c", relative_pos[:, k].astype(np.float64) * box_size / r200c,
                      host_ok & m200c_ok, relative_pos_ok[:, k])
            table.set(f"host_relative_velocity_{axis}_v200c", relative_vel[:, k].astype(np.float64) * 1000.0 / v200c,
                      host_ok & m200c_ok, relative_vel_ok[:, k])

        # Offset between the group's potential minimum and its centre of mass. It is first formed over the
        # catalog's R200c (stored in float32), then re-expressed over the R200c of the jittered M200c.
        group_pos, _ = _host_field(groups, "GroupPos", host_id, host_ok, slice(0, 3))
        group_cm, _ = _host_field(groups, "GroupCM", host_id, host_ok, slice(0, 3))
        native_r200c, _ = _host_field(groups, "Group_R_Crit200", host_id, host_ok)
        group_offset_ok = (host_ok & np.isfinite(group_pos).all(axis=1) & np.isfinite(group_cm).all(axis=1)
                           & np.isfinite(native_r200c) & (native_r200c > 0.0))
        stored_offset = _log1p_ratio_float32(_periodic_distance(group_pos, group_cm, box_size), native_r200c,
                                             group_offset_ok)
        log_native_r200c, native_r200c_ok = _log10_float32(native_r200c, 1.0e-6)
        usable = m200c_ok & host_ok & native_r200c_ok
        offset_ckpc_per_h = (np.expm1(np.where(usable, stored_offset, np.float32(0.0)).astype(np.float64))
                        * np.power(10.0, np.where(usable, log_native_r200c, np.float32(0.0)).astype(np.float64)))
        table.set("log1p_periodic_group_pos_cm_offset_over_r200c", np.log1p(offset_ckpc_per_h / r200c), usable,
                  group_offset_ok)

        if nbody:
            # FoF dark-matter mass; it equals the total FoF mass in an N-body run. Baryon columns stay unavailable.
            fof_dm_mass, fof_dm_mass_ok = host_mass["log10_host_dm_mass_1e10_msun_per_h"]
            table.set("log10_host_fof_mass_1e10_msun_per_h", fof_dm_mass, host_ok, fof_dm_mass_ok)
        else:
            _read_baryons(table, subhalos, groups, rows, host_id, host_ok, host_jitter)

    return Catalog(suite=suite, sim=sim, nbody=nbody, box_size_mpc_h=box_size / 1000.0, row_index=rows,
                   host_id=host_id, values=table.values, available=table.available, valid=table.valid)


def _read_baryons(table: _Table, subhalos: h5py.Group, groups: h5py.Group, rows: np.ndarray,
                  host_id: np.ndarray, host_ok: np.ndarray, host_jitter: np.ndarray) -> None:
    """Stars, gas, star formation, metals and black holes of the subhalo and of its host FoF group (hydro only)."""

    def subhalo(name: str, component: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw, present = _subhalo_field(subhalos, name, rows, component)
        available = np.full(len(rows), present)
        return raw, available, available & np.isfinite(raw) & (raw >= 0.0)

    stellar, stellar_available, stellar_defined = subhalo("SubhaloMassType", 4)
    gas, gas_available, gas_defined = subhalo("SubhaloMassType", 0)
    table.set_log_pair("log10_positive_stellar_mass_1e10_msun_per_h", "stellar_mass_is_positive",
                       stellar, stellar_available, stellar_defined)
    table.set_log_pair("log10_positive_gas_mass_1e10_msun_per_h", "gas_mass_is_positive", gas, gas_available,
                       gas_defined)
    for name, bit_name, field in (
        ("log10_positive_star_formation_rate_msun_per_yr", "star_formation_rate_is_positive", "SubhaloSFR"),
        ("log10_positive_sfr_within_rad_msun_per_yr", "sfr_within_rad_is_positive", "SubhaloSFRinRad"),
        ("log10_positive_black_hole_mass_1e10_msun_per_h", "black_hole_mass_is_positive", "SubhaloBHMass"),
        ("log10_positive_black_hole_accretion_rate_1e10_msun_per_0.978_gyr",
         "black_hole_accretion_rate_is_positive", "SubhaloBHMdot"),
        ("log10_positive_sfr_weighted_gas_metallicity", "sfr_weighted_gas_metallicity_is_positive",
         "SubhaloGasMetallicitySfr"),
    ):
        table.set_log_pair(name, bit_name, *subhalo(field))
    # A metallicity is defined only where the parent component exists.
    raw, available, defined = subhalo("SubhaloGasMetallicity")
    table.set_log_pair("log10_positive_gas_metallicity_metal_mass_fraction", "gas_metallicity_is_positive",
                       raw, available, defined & gas_available & np.isfinite(gas) & (gas > 0.0))
    raw, available, defined = subhalo("SubhaloStarMetallicity")
    table.set_log_pair("log10_positive_stellar_metallicity_metal_mass_fraction", "stellar_metallicity_is_positive",
                       raw, available, defined & stellar_available & np.isfinite(stellar) & (stellar > 0.0))
    raw, available, defined = subhalo("SubhaloHalfmassRadType", 4)
    table.set_log_pair("log10_positive_stellar_halfmass_radius_ckpc_per_h", None, raw, available, defined)

    # Host FoF group. Its masses are expressed as fractions of the jittered total FoF mass
    # M_FoF' = sum_k GroupMassType[host, k] * 10^jitter, so the jitter reaches them only through M_FoF'.
    mass_type, _ = _host_field(groups, "GroupMassType", host_id, host_ok, slice(0, 6))
    total = mass_type.astype(np.float64).sum(axis=1)
    defined = (host_ok & np.isfinite(mass_type).all(axis=1) & (mass_type >= 0.0).all(axis=1)
               & np.isfinite(total) & (total > 0.0))
    log_total = np.log10(np.where(defined, total, 1.0)) + host_jitter.astype(np.float64)
    table.set("log10_host_fof_mass_1e10_msun_per_h", log_total, host_ok, defined)
    table.set_log_pair("log10_positive_host_fof_stellar_fraction", "host_fof_stellar_fraction_is_positive",
                       mass_type[:, 4], defined, defined, log_total)
    table.set_log_pair("log10_positive_host_fof_gas_fraction", "host_fof_gas_fraction_is_positive",
                       mass_type[:, 0], defined, defined, log_total)
    for name, bit_name, field, log_offset in (
        ("log10_positive_host_fof_black_hole_fraction", "host_fof_black_hole_fraction_is_positive",
         "GroupBHMass", log_total),
        ("log10_positive_host_group_sfr_msun_per_yr", "host_group_sfr_is_positive", "GroupSFR", 0.0),
    ):
        raw, _ = _host_field(groups, field, host_id, host_ok)
        field_defined = host_ok & np.isfinite(raw) & (raw >= 0.0)
        table.set_log_pair(name, bit_name, raw, field_defined, field_defined, log_offset)

    # Dark-matter fraction of the host, 1 - (non-DM mass) / M_FoF', written as log1p(-x) / ln 10 so that it
    # keeps full precision when the fraction is close to 1.
    non_dm = mass_type.astype(np.float64)[:, [0, 2, 3, 4, 5]].sum(axis=1)
    ratio = np.where(defined, non_dm / np.where(defined, total, 1.0) * np.power(10.0, -host_jitter.astype(
        np.float64)), 0.0)
    fraction_ok = defined & np.isfinite(1.0 - ratio) & (1.0 - ratio > 0.0)
    table.set("log10_host_fof_dm_fraction", np.log1p(-np.where(fraction_ok, ratio, 0.0)) / math.log(10.0),
              defined, fraction_ok)
