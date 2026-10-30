"""The two pretraining losses.

* ``masked_property_loss`` (M): the student reconstructs the properties hidden by ``masking.sample_masks``.
  Binary columns use a cross-entropy on the raw 0/1 value, all others a smooth L1 loss on the normalised value.
* ``teacher_prediction_loss`` (O): for the scored rows, the student predicts the per-subhalo representation that
  the EMA teacher computes from the unmasked catalog (cosine distance).

The total loss is ``M + 0.3 * O``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from celia.features import BINARY_COLUMNS, NUM_COLUMNS, TOKEN_GROUPS, columns

TEACHER_PREDICTION_WEIGHT = 0.3

# Columns are averaged into these groups first, so that a group with many columns (or many hidden entries) does
# not dominate M. Positions and host-relative coordinates are never scored, so they have no group here.
LOSS_GROUPS: dict[str, tuple[int, ...]] = {
    "velocity": TOKEN_GROUPS["velocity"],
    "host_relative_velocity": TOKEN_GROUPS["host_relative_velocity"],
    "dm_mass": TOKEN_GROUPS["dm_mass"],
    "dm_halfmass_radius": columns("log10_dm_halfmass_radius_ckpc_per_h"),
    "dm_dynamics": TOKEN_GROUPS["dynamics"] + columns("log10_subhalo_total_halfmass_radius_ckpc_per_h"),
    "center_offset": TOKEN_GROUPS["center_offset"],
    "host_mass": TOKEN_GROUPS["host_mass"],
    "host_m200c": TOKEN_GROUPS["host_m200c"],
    "host_m500c": TOKEN_GROUPS["host_m500c"],
    "host_center_offset": TOKEN_GROUPS["host_center_offset"],
    "stellar_mass": columns("log10_positive_stellar_mass_1e10_msun_per_h", "stellar_mass_is_positive"),
    "gas_mass": columns("log10_positive_gas_mass_1e10_msun_per_h", "gas_mass_is_positive"),
    "star_formation_rate": columns(
        "log10_positive_star_formation_rate_msun_per_yr", "star_formation_rate_is_positive"
    ),
    "gas_observables": columns(
        "log10_positive_sfr_weighted_gas_metallicity",
        "sfr_weighted_gas_metallicity_is_positive",
        "log10_positive_sfr_within_rad_msun_per_yr",
        "sfr_within_rad_is_positive",
    ),
    "gas_metallicity": TOKEN_GROUPS["gas_metallicity"],
    "stellar_metallicity": TOKEN_GROUPS["stellar_metallicity"],
    "stellar_halfmass_radius": TOKEN_GROUPS["stellar_halfmass_radius"],
    "black_hole_mass": columns("log10_positive_black_hole_mass_1e10_msun_per_h", "black_hole_mass_is_positive"),
    "black_hole_accretion_rate": TOKEN_GROUPS["black_hole_accretion"],
    "host_baryon": columns(
        "log10_positive_host_fof_gas_fraction",
        "host_fof_gas_fraction_is_positive",
        "log10_positive_host_fof_stellar_fraction",
        "host_fof_stellar_fraction_is_positive",
        "log10_positive_host_fof_black_hole_fraction",
        "host_fof_black_hole_fraction_is_positive",
        "log10_positive_host_group_sfr_msun_per_yr",
        "host_group_sfr_is_positive",
        "log10_host_fof_dm_fraction",
    ),
}

_IS_BINARY = torch.zeros(NUM_COLUMNS, dtype=torch.bool)
_IS_BINARY[list(BINARY_COLUMNS)] = True


def masked_property_loss(
    predictions: torch.Tensor, normalized: torch.Tensor, raw: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """M: mean over loss groups of the mean over columns of the mean loss over the hidden entries.

    All tensors are [B, N, F]; ``targets`` marks the scored entries. Means are taken over the whole batch, so a
    column's loss is averaged over all catalogs (hydro and N-body together). Computed in float32.
    """
    predictions = predictions.float()
    is_binary = _IS_BINARY.to(predictions.device)
    per_entry = torch.where(
        is_binary,
        F.binary_cross_entropy_with_logits(predictions, raw.float(), reduction="none"),
        F.smooth_l1_loss(predictions, normalized.float(), reduction="none", beta=1.0),
    )
    count = targets.sum((0, 1))  # [F]
    column_loss = (per_entry * targets).sum((0, 1)) / count.clamp_min(1)

    group_losses = []
    for cols in LOSS_GROUPS.values():
        scored = count[list(cols)] > 0
        if scored.any():
            group_losses.append(column_loss[list(cols)][scored].mean())
    return torch.stack(group_losses).mean()


def teacher_prediction_loss(
    predictions: torch.Tensor, teacher_features: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """O: mean cosine distance between the student's prediction and the teacher's features over ``rows``.

    ``predictions`` and ``teacher_features`` are [B, N, D]; ``rows`` is [B, N].
    """
    cosine = F.cosine_similarity(predictions.float(), teacher_features.float().detach(), dim=-1, eps=1e-12)
    return (1.0 - cosine)[rows].mean()
