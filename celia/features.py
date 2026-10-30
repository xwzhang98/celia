"""The 53 subhalo properties Celia reads, and how they are grouped.

Every subhalo of a catalog becomes one row of 53 numbers. A row also carries two
boolean flags per column:

* ``available``: the simulation provides this field (for example, N-body runs have no gas or stars);
* ``valid``: the value is physically defined for this subhalo (for example, a log mass needs a positive mass).

The columns are organised in 22 *token groups*. The model embeds each group with its own linear map, and masking
hides whole groups. Groups are listed in the order the tokenizer adds them.

Units are part of the column names: ``1e10_msun_per_h`` = 1e10 Msun/h, ``ckpc_per_h`` = comoving kpc/h,
``1000_km_per_s`` = 1000 km/s, ``msun_per_yr`` = Msun/yr, ``box`` = fraction of the box side, ``r200c`` / ``v200c``
= in units of the host's R200c / V200c.
"""

from __future__ import annotations

COLUMNS: tuple[str, ...] = (
    # position, velocity and position/velocity relative to the host halo
    "position_x_box",
    "position_y_box",
    "position_z_box",
    "velocity_x_1000_km_per_s",
    "velocity_y_1000_km_per_s",
    "velocity_z_1000_km_per_s",
    "host_relative_velocity_x_v200c",
    "host_relative_velocity_y_v200c",
    "host_relative_velocity_z_v200c",
    "host_relative_position_x_r200c",
    "host_relative_position_y_r200c",
    "host_relative_position_z_r200c",
    "is_central",
    # dark-matter properties of the subhalo
    "log10_dm_mass_1e10_msun_per_h",
    "log10_dm_halfmass_radius_ckpc_per_h",
    "log10_subhalo_total_halfmass_radius_ckpc_per_h",
    "log10_subhalo_vmax_km_per_s",
    "log10_subhalo_velocity_dispersion_km_per_s",
    "log10_subhalo_vmax_radius_ckpc_per_h",
    "log10_subhalo_spin_norm",
    "log1p_periodic_pos_cm_offset_over_total_halfmass_radius",
    # host halo (FoF group) properties
    "log10_host_fof_mass_1e10_msun_per_h",
    "log10_host_m200c_1e10_msun_per_h",
    "log10_host_m500c_1e10_msun_per_h",
    "log1p_periodic_group_pos_cm_offset_over_r200c",
    # baryonic properties (absent in N-body runs)
    "log10_positive_stellar_mass_1e10_msun_per_h",
    "stellar_mass_is_positive",
    "log10_positive_host_fof_stellar_fraction",
    "host_fof_stellar_fraction_is_positive",
    "log10_positive_gas_mass_1e10_msun_per_h",
    "gas_mass_is_positive",
    "log10_positive_host_fof_gas_fraction",
    "host_fof_gas_fraction_is_positive",
    "log10_positive_star_formation_rate_msun_per_yr",
    "star_formation_rate_is_positive",
    "log10_positive_sfr_within_rad_msun_per_yr",
    "sfr_within_rad_is_positive",
    "log10_positive_host_group_sfr_msun_per_yr",
    "host_group_sfr_is_positive",
    "log10_positive_gas_metallicity_metal_mass_fraction",
    "gas_metallicity_is_positive",
    "log10_positive_stellar_metallicity_metal_mass_fraction",
    "stellar_metallicity_is_positive",
    "log10_positive_stellar_halfmass_radius_ckpc_per_h",
    "log10_positive_sfr_weighted_gas_metallicity",
    "sfr_weighted_gas_metallicity_is_positive",
    "log10_positive_black_hole_mass_1e10_msun_per_h",
    "black_hole_mass_is_positive",
    "log10_positive_host_fof_black_hole_fraction",
    "host_fof_black_hole_fraction_is_positive",
    "log10_positive_black_hole_accretion_rate_1e10_msun_per_0.978_gyr",
    "black_hole_accretion_rate_is_positive",
    "log10_host_fof_dm_fraction",
)
NUM_COLUMNS = len(COLUMNS)
COLUMN_INDEX = {name: i for i, name in enumerate(COLUMNS)}


def columns(*names: str) -> tuple[int, ...]:
    """Column indices of the given column names."""
    return tuple(COLUMN_INDEX[name] for name in names)


def column_range(first: str, last: str) -> tuple[int, ...]:
    """Indices of all columns from ``first`` to ``last``, inclusive."""
    return tuple(range(COLUMN_INDEX[first], COLUMN_INDEX[last] + 1))


TOKEN_GROUPS: dict[str, tuple[int, ...]] = {
    "position": column_range("position_x_box", "position_z_box"),
    "velocity": column_range("velocity_x_1000_km_per_s", "velocity_z_1000_km_per_s"),
    "host_relative_velocity": column_range("host_relative_velocity_x_v200c", "host_relative_velocity_z_v200c"),
    "host_relative_position": column_range("host_relative_position_x_r200c", "is_central"),
    "dm_mass": columns("log10_dm_mass_1e10_msun_per_h"),
    "halfmass_radius": columns("log10_dm_halfmass_radius_ckpc_per_h", "log10_subhalo_total_halfmass_radius_ckpc_per_h"),
    "dynamics": column_range("log10_subhalo_vmax_km_per_s", "log10_subhalo_spin_norm"),
    "center_offset": columns("log1p_periodic_pos_cm_offset_over_total_halfmass_radius"),
    "host_mass": columns("log10_host_fof_mass_1e10_msun_per_h"),
    "host_m200c": columns("log10_host_m200c_1e10_msun_per_h"),
    "host_m500c": columns("log10_host_m500c_1e10_msun_per_h"),
    "host_center_offset": columns("log1p_periodic_group_pos_cm_offset_over_r200c"),
    "stellar": column_range("log10_positive_stellar_mass_1e10_msun_per_h", "host_fof_stellar_fraction_is_positive"),
    "gas": column_range("log10_positive_gas_mass_1e10_msun_per_h", "host_fof_gas_fraction_is_positive"),
    "star_formation": column_range("log10_positive_star_formation_rate_msun_per_yr", "host_group_sfr_is_positive"),
    "gas_metallicity": column_range(
        "log10_positive_gas_metallicity_metal_mass_fraction", "gas_metallicity_is_positive"
    ),
    "stellar_metallicity": column_range(
        "log10_positive_stellar_metallicity_metal_mass_fraction", "stellar_metallicity_is_positive"
    ),
    "stellar_halfmass_radius": columns("log10_positive_stellar_halfmass_radius_ckpc_per_h"),
    "sf_gas_metallicity": column_range(
        "log10_positive_sfr_weighted_gas_metallicity", "sfr_weighted_gas_metallicity_is_positive"
    ),
    "black_hole": column_range(
        "log10_positive_black_hole_mass_1e10_msun_per_h", "host_fof_black_hole_fraction_is_positive"
    ),
    "black_hole_accretion": column_range(
        "log10_positive_black_hole_accretion_rate_1e10_msun_per_0.978_gyr",
        "black_hole_accretion_rate_is_positive",
    ),
    "host_dm_fraction": columns("log10_host_fof_dm_fraction"),
}

# The position group is used only for the spatial layers (voxel context and dense attention); it is not a
# content token and is never hidden.
CONTENT_GROUPS: tuple[str, ...] = tuple(g for g in TOKEN_GROUPS if g != "position")

# 3-vectors: the random cube rotation acts on them, and each shares one standard deviation in normalisation.
VECTORS: dict[str, tuple[int, ...]] = {
    "position": TOKEN_GROUPS["position"],
    "velocity": TOKEN_GROUPS["velocity"],
    "host_relative_velocity": TOKEN_GROUPS["host_relative_velocity"],
    "host_relative_position": TOKEN_GROUPS["host_relative_position"][:3],
}

# 0/1 columns: predicted with a binary cross-entropy instead of a smooth L1 loss.
BINARY_COLUMNS: tuple[int, ...] = tuple(
    i for i, name in enumerate(COLUMNS) if name == "is_central" or name.endswith("_is_positive")
)

# Columns that describe the host halo rather than the subhalo itself. When a subhalo is masked as an object,
# these columns are hidden on every subhalo of the same host, so they cannot be copied from a neighbour.
HOST_COLUMNS: tuple[int, ...] = (
    TOKEN_GROUPS["host_mass"]
    + TOKEN_GROUPS["host_m200c"]
    + TOKEN_GROUPS["host_m500c"]
    + TOKEN_GROUPS["host_center_offset"]
    + columns(
        "log10_positive_host_fof_stellar_fraction",
        "host_fof_stellar_fraction_is_positive",
        "log10_positive_host_fof_gas_fraction",
        "host_fof_gas_fraction_is_positive",
        "log10_positive_host_group_sfr_msun_per_yr",
        "host_group_sfr_is_positive",
        "log10_positive_host_fof_black_hole_fraction",
        "host_fof_black_hole_fraction_is_positive",
        "log10_host_fof_dm_fraction",
    )
)

# Baryonic columns: absent in N-body runs, and hidden together in "hydro dropout".
BARYON_COLUMNS: tuple[int, ...] = column_range(
    "log10_positive_stellar_mass_1e10_msun_per_h", "log10_host_fof_dm_fraction"
)
DM_COLUMNS: tuple[int, ...] = tuple(i for i in range(NUM_COLUMNS) if i not in BARYON_COLUMNS)
