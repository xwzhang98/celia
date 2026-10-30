"""Random cube symmetries and the masking of subhalo properties used in pretraining.

For every catalog of a training batch, in this order:

1. ``augment_symmetries`` rotates the catalog by one of the 24 proper rotations of the cube and shifts it
   periodically. The box is periodic and cubic, so these maps produce an equally valid catalog.
2. ``sample_masks`` decides what the student sees:

   * *hydro dropout*: with probability 0.25 all baryonic properties are hidden;
   * *group masking*: each token group is hidden in the whole catalog with its own probability (0.5 for dark-matter
     groups, 1/3 for the four baryonic parent groups; positions and host-relative coordinates are never hidden);
   * *object masking*: each host halo is selected with probability 0.2 and every property of its subhalos except
     position and host-relative position is hidden;
   * deterministic rules that hide properties which would otherwise reveal a hidden one (host properties are
     shared by all subhalos of a host; the host's DM fraction follows from its gas, stellar and BH content; the
     host-relative coordinates are in units of R200c and V200c, which follow from M200c).

   Hidden entries that the catalog actually contains become the targets of the masked-property loss.

Masks are drawn from one ``torch.Generator`` per update, so a run is reproducible for a fixed seed and device.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from celia.features import BARYON_COLUMNS, HOST_COLUMNS, TOKEN_GROUPS, VECTORS, columns

HYDRO_DROPOUT_PROBABILITY = 0.25
OBJECT_MASK_PROBABILITY = 0.2

# Probability of hiding each token group in a whole catalog. Groups not listed here are never sampled on their
# own: they are either never hidden, follow a parent group, or follow a deterministic rule below.
GROUP_MASK_PROBABILITY: dict[str, float] = {
    "velocity": 0.5,
    "dm_mass": 0.5,
    "halfmass_radius": 0.5,
    "dynamics": 0.5,
    "center_offset": 0.5,
    "host_mass": 0.5,
    "host_center_offset": 0.5,
    "stellar": 1 / 3,
    "gas": 1 / 3,
    "star_formation": 1 / 3,
    "black_hole": 1 / 3,
}

# A child group is hidden whenever its parent is (for example, gas metallicity is meaningless without gas).
PARENT_GROUP: dict[str, str] = {
    "host_relative_velocity": "velocity",
    "host_m200c": "host_mass",
    "host_m500c": "host_mass",
    "gas_metallicity": "gas",
    "stellar_metallicity": "stellar",
    "stellar_halfmass_radius": "stellar",
    "sf_gas_metallicity": "star_formation",
    "black_hole_accretion": "black_hole",
}

# Groups that object masking leaves visible: where the subhalo is, and where it sits in its host.
KEPT_UNDER_OBJECT_MASK = TOKEN_GROUPS["position"] + TOKEN_GROUPS["host_relative_position"]

# Groups that are inputs only and never scored by the masked-property loss.
NEVER_TARGETS = TOKEN_GROUPS["position"] + TOKEN_GROUPS["host_relative_position"]

# If any of these host masses is hidden, R200c and V200c are unknown, so the coordinates expressed in those units
# are hidden on every subhalo of the host.
R200C_TRIGGERS = TOKEN_GROUPS["host_mass"] + TOKEN_GROUPS["host_m200c"] + TOKEN_GROUPS["host_m500c"]
R200C_DEPENDENT = (
    TOKEN_GROUPS["host_relative_position"]
    + TOKEN_GROUPS["host_relative_velocity"]
    + TOKEN_GROUPS["host_center_offset"]
)

# The host DM fraction is one minus the host gas, star and black-hole fractions.
HOST_DM_FRACTION = TOKEN_GROUPS["host_dm_fraction"]
HOST_DM_FRACTION_SOURCES = TOKEN_GROUPS["stellar"] + TOKEN_GROUPS["gas"] + TOKEN_GROUPS["black_hole"]

# (targets, triggers): a target is not scored when a visible trigger bit equals 1, because the trigger then
# implies the target trivially (for example, a positive gas metallicity implies gas, hence a positive gas mass).
LOSS_SHIELDS: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
    (
        columns("stellar_mass_is_positive", "host_fof_stellar_fraction_is_positive"),
        columns("gas_metallicity_is_positive", "sfr_within_rad_is_positive"),
    ),
    (columns("gas_mass_is_positive"), columns("star_formation_rate_is_positive")),
    (
        columns("host_fof_gas_fraction_is_positive"),
        columns("star_formation_rate_is_positive", "host_group_sfr_is_positive"),
    ),
)


def _proper_cube_rotations() -> torch.Tensor:
    """The 24 rotation matrices that map the cube onto itself (signed permutations with determinant +1)."""
    rotations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = torch.zeros(3, 3)
            for row, (col, sign) in enumerate(zip(perm, signs)):
                matrix[row, col] = sign
            if torch.det(matrix) > 0:
                rotations.append(matrix)
    return torch.stack(rotations)


CUBE_ROTATIONS = _proper_cube_rotations()


def augment_symmetries(
    values: torch.Tensor,
    available: torch.Tensor,
    valid: torch.Tensor,
    padding: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate every catalog by a random cube rotation and shift its positions by a random periodic offset.

    ``values`` are raw (not normalised); positions are box fractions in [0, 1). Rotations act on the four 3-vectors
    and on their per-component flags. Padding rows are left untouched.
    """
    batch = values.shape[0]
    device = values.device
    choice = torch.randint(len(CUBE_ROTATIONS), (batch,), generator=generator, device=device)
    rotation = CUBE_ROTATIONS.to(device)[choice]  # [B, 3, 3]
    shift = torch.rand(batch, 1, 3, generator=generator, device=device)
    # A signed permutation: output axis k reads input axis source[k] with sign sign[k].
    source = rotation.abs().argmax(-1)  # [B, 3]
    sign = rotation.sum(-1)  # [B, 3]

    values, available, valid = values.clone(), available.clone(), valid.clone()
    real = ~padding[..., None]
    for name, cols in VECTORS.items():
        cols_t = torch.tensor(cols, device=device)
        index = cols_t[source][:, None, :].expand(-1, values.shape[1], -1)  # [B, N, 3]
        rotated = values.gather(-1, index) * sign[:, None, :]
        rotated_available = available.gather(-1, index)
        rotated_valid = valid.gather(-1, index)
        if name == "position":
            rotated = torch.where(rotated_available, torch.remainder(rotated + shift, 1.0), rotated)
        values[..., cols] = torch.where(real, rotated, values[..., cols])
        available[..., cols] = torch.where(real, rotated_available, available[..., cols])
        valid[..., cols] = torch.where(real, rotated_valid, valid[..., cols])
    return values, available, valid


@dataclass
class MaskedView:
    """What the student is shown, and what it is asked to predict."""

    observed: torch.Tensor  # [B, N, F] the student sees these entries
    student_available: torch.Tensor  # [B, N, F] hidden entries look like natively missing data to the student
    student_valid: torch.Tensor  # [B, N, F]
    targets: torch.Tensor  # [B, N, F] entries scored by the masked-property loss
    predictor_rows: torch.Tensor  # [B, N] rows scored by the teacher-prediction loss


def _host_keys(host_id: torch.Tensor) -> torch.Tensor:
    """One integer per row that is equal exactly for rows of the same host in the same catalog.

    Rows without a host (host_id < 0) get a key of their own.
    """
    batch, rows = host_id.shape
    catalog = torch.arange(batch, device=host_id.device)[:, None]
    own_row = torch.arange(rows, device=host_id.device)[None, :]
    key = torch.where(host_id >= 0, host_id, -1 - own_row)
    _, inverse = torch.unique(torch.stack([catalog.expand_as(key), key], -1).reshape(-1, 2), dim=0, return_inverse=True)
    return inverse.view(batch, rows)


def _spread_over_hosts(flag: torch.Tensor, host_key: torch.Tensor) -> torch.Tensor:
    """True on every row whose host has at least one row with ``flag`` set."""
    per_host = torch.zeros(int(host_key.max()) + 1, dtype=torch.bool, device=flag.device)
    per_host.index_put_((host_key[flag],), torch.ones((), dtype=torch.bool, device=flag.device))
    return per_host[host_key]


def _any_hidden(usable: torch.Tensor, observed: torch.Tensor, cols: tuple[int, ...]) -> torch.Tensor:
    return (usable[..., cols] & ~observed[..., cols]).any(-1)


def sample_masks(
    raw_values: torch.Tensor,
    available: torch.Tensor,
    valid: torch.Tensor,
    padding: torch.Tensor,
    host_id: torch.Tensor,
    generator: torch.Generator,
) -> MaskedView:
    """Draw the masks of one training batch (see the module docstring for the rules).

    ``raw_values`` are needed only for the 0/1 bits of the loss shields.
    """
    batch = available.shape[0]
    device = available.device
    usable = available & valid & ~padding[..., None]

    # 1. Hydro dropout: hide all baryonic properties of the catalog.
    hydro_dropped = torch.rand(batch, generator=generator, device=device) < HYDRO_DROPOUT_PROBABILITY
    maskable = usable.clone()
    maskable[..., BARYON_COLUMNS] &= ~hydro_dropped[:, None, None]

    # 2. Group masking, one decision per catalog and group; children follow their parents.
    keep_group: dict[str, torch.Tensor] = {}
    for group, probability in GROUP_MASK_PROBABILITY.items():
        keep_group[group] = torch.rand(batch, generator=generator, device=device) >= probability
    for child, parent in PARENT_GROUP.items():
        keep_group[child] = keep_group[parent]
    observed = maskable.clone()
    for group, keep in keep_group.items():
        observed[..., TOKEN_GROUPS[group]] &= keep[:, None, None]

    # 3. Object masking: select whole host halos.
    host_key = _host_keys(host_id)
    host_selected = torch.rand(int(host_key.max()) + 1, generator=generator, device=device) < OBJECT_MASK_PROBABILITY
    object_masked = host_selected[host_key] & ~padding
    observed &= ~object_masked[..., None]
    observed[..., KEPT_UNDER_OBJECT_MASK] |= usable[..., KEPT_UNDER_OBJECT_MASK] & object_masked[..., None]

    # 4. Deterministic rules.
    shares_masked_host = _spread_over_hosts(object_masked, host_key)
    observed[..., HOST_COLUMNS] &= ~shares_masked_host[..., None]
    observed[..., HOST_DM_FRACTION] &= ~_any_hidden(usable, observed, HOST_DM_FRACTION_SOURCES)[..., None]
    r200c_unknown = _spread_over_hosts(_any_hidden(usable, observed, R200C_TRIGGERS) & ~padding, host_key)
    observed[..., R200C_DEPENDENT] &= ~r200c_unknown[..., None]
    observed &= ~padding[..., None]

    # 5. Targets of the masked-property loss.
    targets = usable & ~observed
    targets[..., NEVER_TARGETS] = False
    for target_cols, trigger_cols in LOSS_SHIELDS:
        fired = (observed[..., trigger_cols] & (raw_values[..., trigger_cols] == 1)).any(-1)
        targets[..., target_cols] &= ~fired[..., None]

    # 6. Rows scored by the teacher-prediction loss: the row must keep some visible content and miss some group.
    group_usable = torch.stack([usable[..., cols].any(-1) for cols in TOKEN_GROUPS.values()], -1)
    group_visible = torch.stack([observed[..., cols].any(-1) for cols in TOKEN_GROUPS.values()], -1)
    is_position = torch.tensor([g == "position" for g in TOKEN_GROUPS], device=device)
    predictor_rows = (
        ~padding
        & (group_usable.sum(-1) >= 2)
        & (group_visible & ~is_position).any(-1)
        & (group_usable & ~group_visible).any(-1)
    )

    hidden = usable & ~observed
    return MaskedView(
        observed=observed,
        student_available=available & ~hidden,
        student_valid=valid & ~hidden,
        targets=targets,
        predictor_rows=predictor_rows,
    )
