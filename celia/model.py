"""The Celia catalog encoder and its two training heads.

A catalog is a set of subhalos, each described by the 53 columns of :mod:`celia.features` plus three boolean
flags per column (available, valid, observed). The encoder turns it into

* ``object_features`` C: one 256-d contextual vector per subhalo (the exported representation), and
* ``latent_slots`` Z: 32 learned summary vectors of the whole catalog.

Pipeline:

1. ``SubhaloTokenizer``: one token per subhalo from its visible property groups and their mask states.
2. ``VoxelContext``: tokens are deposited on a periodic 16^3 grid, mixed by two circular 3-D convolutions and read
   back, giving each subhalo its large-scale environment.
3. ``PeriodicAttention``: one dense attention layer over all subhalos with a learned bias on periodic distance.
4. Four ``InducedSetBlock`` layers (Set Transformer ISAB, 64 inducing points) and a final LayerNorm -> C.
5. Pooling by attention onto 32 learned queries, slot self-attention and a LayerNorm -> Z.

Two heads are used only in training: ``PropertyDecoder`` predicts the hidden properties from C (the masked-property
loss), and ``TeacherPredictor`` predicts the EMA teacher's C of a subhalo from its visible token, reading only the
latent slots Z (the teacher-prediction loss).

Padding rows (``padding`` True) are exact zeros after every block. Hidden values never enter a linear layer: they
are replaced by zeros with ``torch.where`` (multiplying by a 0/1 mask would turn a NaN into a NaN).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from celia.features import CONTENT_GROUPS, NUM_COLUMNS, TOKEN_GROUPS


@dataclass(frozen=True)
class ModelConfig:
    """Sizes of the encoder (the paper model's values are the defaults)."""

    dim: int = 256
    heads: int = 8
    inducing_points: int = 64
    isab_blocks: int = 4
    latent_slots: int = 32
    voxel_resolution: int = 16
    voxel_channels: int = 32
    voxel_conv_blocks: int = 2
    distance_bias_hidden: int = 16
    activation_checkpointing: bool = True


@dataclass
class EncoderOutput:
    """What the encoder returns for a batch of B catalogs padded to N rows."""

    object_features: torch.Tensor  # C [B, N, dim], zeros on padding rows
    latent_slots: torch.Tensor  # Z [B, latent_slots, dim]


def feed_forward(dim: int) -> nn.Sequential:
    """The position-wise MLP dim -> 4 dim -> dim used after every attention."""
    return nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))


def zero_rows(x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Set the vectors of the rows where ``rows`` is True to exactly zero."""
    return x.masked_fill(rows.unsqueeze(-1), 0.0)


class SubhaloTokenizer(nn.Module):
    """One token per subhalo, summed over its content groups (all groups except position).

    Each group g adds ``property_embedding[g](visible values)`` (no bias, so a fully hidden group adds nothing
    through this path) and ``mask_state_embedding[g]`` of the fractions of the group's columns that are available,
    valid, observed and readable (= all three, on a real row). The state term lets the model tell a property that
    does not exist (N-body run, undefined value) from one that was hidden by the masking.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.groups = {name: TOKEN_GROUPS[name] for name in CONTENT_GROUPS}
        self.property_embedding = nn.ModuleDict(
            {name: nn.Linear(len(cols), dim, bias=False) for name, cols in self.groups.items()}
        )
        self.mask_state_embedding = nn.ModuleDict({name: nn.Linear(4, dim) for name in self.groups})
        self.base_token = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(
        self,
        values: torch.Tensor,
        available: torch.Tensor,
        valid: torch.Tensor,
        observed: torch.Tensor,
        padding: torch.Tensor,
    ) -> torch.Tensor:
        readable = available & valid & observed & ~padding.unsqueeze(-1)
        visible_values = torch.where(readable, values, torch.zeros_like(values))
        token = self.base_token.expand(*values.shape[:2], -1)
        for name, cols in self.groups.items():
            index = torch.as_tensor(cols, device=values.device)
            flags = (available, valid, observed, readable)
            state = torch.stack([m.index_select(-1, index).to(values.dtype).mean(dim=-1) for m in flags], dim=-1)
            token = (
                token
                + self.property_embedding[name](visible_values.index_select(-1, index))
                + self.mask_state_embedding[name](state)
            )
        token = self.norm(token)
        token = token + self.mlp(token)
        return zero_rows(token, padding)


class MaskedAttention(nn.Module):
    """Multi-head attention that ignores masked keys and outputs zeros for masked queries.

    Masked logits are set to the most negative finite number (not -inf, which would give NaN when every key is
    masked); the softmax weights are then re-zeroed on masked keys and renormalised, so a query with no valid key
    receives exactly zero.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor
    ) -> torch.Tensor:
        q = self.split_heads(self.q_proj(query))
        k = self.split_heads(self.k_proj(key_value))
        v = self.split_heads(self.v_proj(key_value))
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.head_dim**-0.5
        key_valid = key_mask[:, None, None, :]
        logits = logits.masked_fill(~key_valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(key_valid, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(query.shape)
        return zero_rows(self.out_proj(attended), ~query_mask)


class AttentionBlock(nn.Module):
    """Pre-norm residual attention of ``query`` over ``key_value``, then a residual MLP."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.key_value_norm = nn.LayerNorm(dim)
        self.attention = MaskedAttention(dim, heads)
        self.ff_norm = nn.LayerNorm(dim)
        self.feed_forward = feed_forward(dim)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor
    ) -> torch.Tensor:
        x = query + self.attention(self.query_norm(query), self.key_value_norm(key_value), query_mask, key_mask)
        x = x + self.feed_forward(self.ff_norm(x))
        return zero_rows(x, ~query_mask)


class VoxelContext(nn.Module):
    """Large-scale environment from a periodic grid: deposit tokens, convolve, read back (cloud-in-cell).

    Each cell holds the weighted mean of the deposited token channels plus log(1 + occupancy), so both what is
    there and how much of it is there enter the convolutions. The token stays on a residual path.
    """

    def __init__(self, dim: int, resolution: int, channels: int, conv_blocks: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.channels = channels
        self.input_norm = nn.LayerNorm(dim)
        self.deposit = nn.Linear(dim, channels)
        self.grid_input = nn.Conv3d(channels + 1, channels, kernel_size=1)
        self.conv_norms = nn.ModuleList([nn.GroupNorm(8, channels) for _ in range(conv_blocks)])
        self.convs = nn.ModuleList([nn.Conv3d(channels, channels, kernel_size=3) for _ in range(conv_blocks)])
        self.read_out = nn.Linear(channels, dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.feed_forward = feed_forward(dim)
        corners = [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        self.register_buffer("corner_offsets", torch.tensor(corners), persistent=False)

    def cic_corners(self, positions: torch.Tensor, has_position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat cell index [B, N, 8] (offset per catalog) and weight [B, N, 8] of the 8 cells around each point."""
        r = self.resolution
        # Wrap again: a position that rounded up to exactly 1.0 must land in cell 0.
        coordinate = torch.remainder(positions, 1.0) * float(r)
        lower = torch.floor(coordinate).long()
        fraction = coordinate - lower.to(coordinate.dtype)
        cells = torch.remainder(lower.unsqueeze(2) + self.corner_offsets, r)  # [B, N, 8, 3]
        upper = self.corner_offsets.bool()
        weight = torch.where(upper, fraction.unsqueeze(2), 1.0 - fraction.unsqueeze(2)).prod(dim=-1)
        weight = weight * has_position.unsqueeze(-1).to(weight.dtype)
        flat = cells[..., 0] * r * r + cells[..., 1] * r + cells[..., 2]
        catalog_offset = torch.arange(positions.shape[0], device=positions.device)[:, None, None] * r**3
        return flat + catalog_offset, weight

    def forward(
        self, tokens: torch.Tensor, positions: torch.Tensor, padding: torch.Tensor, has_position: torch.Tensor
    ) -> torch.Tensor:
        batch, r, c = tokens.shape[0], self.resolution, self.channels
        cell, weight = self.cic_corners(positions, has_position)
        content = self.deposit(self.input_norm(tokens))
        deposited = tokens.new_zeros((batch * r**3, c))
        deposited.index_add_(0, cell.reshape(-1), (content.unsqueeze(2) * weight.unsqueeze(-1)).reshape(-1, c))
        occupancy = tokens.new_zeros((batch * r**3, 1))
        occupancy.index_add_(0, cell.reshape(-1), weight.reshape(-1, 1))
        mean_content = deposited / occupancy.clamp_min(torch.finfo(deposited.dtype).eps)
        grid = torch.cat((mean_content, torch.log1p(occupancy)), dim=-1).reshape(batch, r, r, r, c + 1)
        grid = self.grid_input(grid.permute(0, 4, 1, 2, 3).contiguous())
        for norm, conv in zip(self.conv_norms, self.convs):
            grid = grid + conv(F.pad(F.gelu(norm(grid)), (1, 1, 1, 1, 1, 1), mode="circular"))
        flat_grid = grid.permute(0, 2, 3, 4, 1).reshape(batch * r**3, c)
        sampled = flat_grid.index_select(0, cell.reshape(-1)).reshape(*cell.shape, c)
        sampled = (sampled * weight.unsqueeze(-1)).sum(dim=2)
        x = tokens + zero_rows(self.read_out(sampled), ~has_position)
        x = x + self.feed_forward(self.ff_norm(x))
        return zero_rows(x, padding)


class PeriodicAttention(nn.Module):
    """Dense attention between all positioned subhalos, biased by their periodic distance.

    The bias of each head is a small MLP of the minimum-image distance r (in box units) seen at three scales: r,
    log(1 + r * box size in Mpc/h) and log(1 + r * particle-grid side), so one set of weights serves boxes of
    different size and resolution. Its last layer starts at zero (plain content attention at initialisation).
    Rows without a position neither attend nor are attended to.
    """

    def __init__(self, dim: int, heads: int, bias_hidden: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.input_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.distance_bias = nn.Sequential(
            nn.Linear(3, bias_hidden), nn.GELU(), nn.Linear(bias_hidden, heads, bias=False)
        )
        nn.init.zeros_(self.distance_bias[-1].weight)
        self.ff_norm = nn.LayerNorm(dim)
        self.feed_forward = feed_forward(dim)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:2], self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        padding: torch.Tensor,
        has_position: torch.Tensor,
        box_size_mpc_h: torch.Tensor,
        particle_grid_side: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.input_norm(tokens)
        positioned = has_position[:, None, :, None].to(tokens.dtype)
        q = self.split_heads(self.q_proj(normed)) * positioned
        k = self.split_heads(self.k_proj(normed)) * positioned
        v = self.split_heads(self.v_proj(normed)) * positioned
        delta = positions[:, :, None, :] - positions[:, None, :, :]
        delta = delta - torch.round(delta)  # minimum image
        r = torch.sqrt(torch.sum(delta * delta, dim=-1).clamp_min(1.0e-12))
        box = box_size_mpc_h.to(tokens.dtype)[:, None, None]
        grid_side = particle_grid_side.to(tokens.dtype)[:, None, None]
        scales = torch.stack((r, torch.log1p(r * box), torch.log1p(r * grid_side)), dim=-1)
        bias = self.distance_bias(scales).permute(0, 3, 1, 2)  # [B, heads, N, N]
        # A catalog with no positioned row keeps key 0, so the softmax has at least one finite logit.
        key_allowed = has_position.clone()
        key_allowed[~key_allowed.any(dim=1), 0] = True
        bias = bias.masked_fill(~key_allowed[:, None, None, :], torch.finfo(bias.dtype).min)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        attended = self.out_proj(attended.transpose(1, 2).reshape(tokens.shape))
        x = tokens + zero_rows(attended, ~has_position)
        x = x + self.feed_forward(self.ff_norm(x))
        return zero_rows(x, padding)


class InducedSetBlock(nn.Module):
    """Set Transformer ISAB: learned inducing points read the catalog, then every subhalo reads them back.

    Cost is O(N x inducing points) instead of O(N^2).
    """

    def __init__(self, dim: int, heads: int, inducing_points: int) -> None:
        super().__init__()
        self.inducing_points = nn.Parameter(torch.empty(1, inducing_points, dim))
        nn.init.trunc_normal_(self.inducing_points, std=0.02)
        self.inducing_from_objects = AttentionBlock(dim, heads)
        self.objects_from_inducing = AttentionBlock(dim, heads)

    def forward(self, objects: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        inducing = self.inducing_points.expand(objects.shape[0], -1, -1)
        all_inducing = torch.ones(inducing.shape[:2], dtype=torch.bool, device=objects.device)
        induced = self.inducing_from_objects(inducing, objects, all_inducing, real)
        return self.objects_from_inducing(objects, induced, real, all_inducing)


class CatalogEncoder(nn.Module):
    """The Celia encoder: subhalo table -> per-subhalo features C and catalog latent slots Z."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        d = config.dim
        self.config = config
        self.position_columns = TOKEN_GROUPS["position"]
        self.tokenizer = SubhaloTokenizer(d)
        self.voxel_context = VoxelContext(d, config.voxel_resolution, config.voxel_channels, config.voxel_conv_blocks)
        self.periodic_attention = PeriodicAttention(d, config.heads, config.distance_bias_hidden)
        self.blocks = nn.ModuleList(
            [InducedSetBlock(d, config.heads, config.inducing_points) for _ in range(config.isab_blocks)]
        )
        self.output_norm = nn.LayerNorm(d)
        self.pool_projection = nn.Linear(d, d)
        self.latent_queries = nn.Parameter(torch.empty(1, config.latent_slots, d))
        nn.init.trunc_normal_(self.latent_queries, std=0.02)
        self.latent_pool = AttentionBlock(d, config.heads)
        self.latent_self_attention = AttentionBlock(d, config.heads)
        self.latent_norm = nn.LayerNorm(d)

    def tokenize(
        self,
        values: torch.Tensor,
        available: torch.Tensor,
        valid: torch.Tensor,
        observed: torch.Tensor,
        padding: torch.Tensor,
    ) -> torch.Tensor:
        """Tokenizer output alone [B, N, dim] (the teacher predictor's queries)."""
        return self.tokenizer(values, available, valid, observed, padding)

    def forward(
        self,
        values: torch.Tensor,
        available: torch.Tensor,
        valid: torch.Tensor,
        observed: torch.Tensor,
        padding: torch.Tensor,
        box_size_mpc_h: torch.Tensor,
        particle_grid_side: torch.Tensor,
    ) -> EncoderOutput:
        """Encode B catalogs.

        Args:
            values: [B, N, 53] normalised columns; positions (columns 0-2) are box fractions.
            available, valid, observed: [B, N, 53] bool flags; hidden entries have ``observed`` False.
            padding: [B, N] bool, True for padding rows.
            box_size_mpc_h, particle_grid_side: [B] box side in Mpc/h and particles per side of the simulation.
        """
        args = (values, available, valid, observed, padding, box_size_mpc_h, particle_grid_side)
        if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
            # Recompute activations in the backward pass: N x N attention at N ~ 1000 dominates memory.
            return checkpoint(self.encode, *args, use_reentrant=False)
        return self.encode(*args)

    def encode(
        self,
        values: torch.Tensor,
        available: torch.Tensor,
        valid: torch.Tensor,
        observed: torch.Tensor,
        padding: torch.Tensor,
        box_size_mpc_h: torch.Tensor,
        particle_grid_side: torch.Tensor,
    ) -> EncoderOutput:
        """The forward pass without activation checkpointing."""
        real = ~padding
        tokens = self.tokenizer(values, available, valid, observed, padding)

        cols = torch.as_tensor(self.position_columns, device=values.device)
        has_position = (available & valid & observed).index_select(-1, cols).all(dim=-1) & real
        positions = values.index_select(-1, cols)
        positions = torch.where(has_position.unsqueeze(-1), positions, torch.zeros_like(positions))
        positions = torch.remainder(positions, 1.0)

        tokens = self.voxel_context(tokens, positions, padding, has_position)
        tokens = self.periodic_attention(
            tokens, positions, padding, has_position, box_size_mpc_h, particle_grid_side
        )
        for block in self.blocks:
            tokens = block(tokens, real)
        object_features = zero_rows(self.output_norm(tokens), padding)

        batch = values.shape[0]
        queries = self.latent_queries.expand(batch, -1, -1)
        all_slots = torch.ones(queries.shape[:2], dtype=torch.bool, device=values.device)
        slots = self.latent_pool(queries, self.pool_projection(object_features), all_slots, real)
        slots = self.latent_self_attention(slots, slots, all_slots, all_slots)
        return EncoderOutput(object_features=object_features, latent_slots=self.latent_norm(slots))


class PropertyDecoder(nn.Module):
    """Masked-property head: C -> 53 predicted columns (values for continuous, logits for 0/1 columns)."""

    def __init__(self, dim: int = 256, num_columns: int = NUM_COLUMNS) -> None:
        super().__init__()
        self.hidden = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, num_columns)

    def forward(self, object_features: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        return self.output(F.gelu(zero_rows(self.hidden(object_features), padding)))


class TeacherPredictor(nn.Module):
    """Teacher-prediction head: a subhalo's visible token attends to the latent slots to predict the teacher's C.

    The query token only chooses attention weights; there is no residual from it, so the prediction must be read
    out of the catalog slots Z.
    """

    def __init__(self, dim: int = 256, heads: int = 8) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.latent_norm = nn.LayerNorm(dim)
        self.attention = MaskedAttention(dim, heads)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, query_tokens: torch.Tensor, latent_slots: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Prediction [B, N, dim] for the rows where ``rows`` is True, zeros elsewhere."""
        all_slots = torch.ones(latent_slots.shape[:2], dtype=torch.bool, device=latent_slots.device)
        out = self.attention(self.query_norm(query_tokens), self.latent_norm(latent_slots), rows, all_slots)
        return zero_rows(self.output_norm(out), ~rows)
