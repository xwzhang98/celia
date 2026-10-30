"""Pretraining loop.

One update:

1. load 32 hydro catalogs of one suite and their 32 N-body twins (``celia.data``);
2. rotate and shift every catalog, then draw the masks (``celia.masking``);
3. the student encodes the masked catalogs; the EMA teacher encodes the unmasked ones;
4. loss = M + 0.3 O (``celia.losses``); AdamW step; the teacher moves towards the student.

Schedule (warmup-stable-decay): the learning rate rises linearly to 2e-4 over 200 updates, stays constant, and
decays linearly to 0 between ``decay_start`` and ``total_updates``. The teacher's EMA momentum is 0.996 until
``decay_start`` and then rises to 1 along a cosine. The paper model: 10,000 updates at constant learning rate, then
a decay to 18,000 updates.

A run directory holds ``config.yaml``, ``step0.pt`` (written by ``scripts/init_model.py``), ``last.pt`` (every 500
updates), ``step<N>.pt`` at the requested stop, and ``log.jsonl`` (one line per update). ``train`` always continues
from the newest of ``last.pt`` and ``step0.pt``, so a run can be stopped and continued at any time.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import yaml

from celia.data import HYDRO_SUITES, PairedCatalogs, SuiteAlternatingSampler, collate_pairs, split_ids
from celia.losses import TEACHER_PREDICTION_WEIGHT, masked_property_loss, teacher_prediction_loss
from celia.masking import augment_symmetries, sample_masks
from celia.model import CatalogEncoder, ModelConfig, PropertyDecoder, TeacherPredictor
from celia.normalization import normalize

SAVE_EVERY = 500


@dataclass
class TrainConfig:
    """Everything that defines a run (``configs/celia.yaml`` holds the paper values)."""

    tables: str  # directory written by scripts/prepare_data.py
    seed: int = 0  # initial weights, views, masks
    split_seed: int = 0  # which simulations are train / val / test; the paper split is 0
    view_cycle: int = 64
    num_workers: int = 8
    peak_learning_rate: float = 2e-4
    warmup_updates: int = 200
    decay_start: int = 10_000
    total_updates: int = 18_000
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 10.0
    ema_momentum: float = 0.996
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config(path: Path) -> TrainConfig:
    """Read a run configuration from YAML (see ``configs/celia.yaml``)."""
    raw = yaml.safe_load(Path(path).read_text())
    raw["model"] = ModelConfig(**raw.get("model", {}))
    if "betas" in raw:
        raw["betas"] = tuple(raw["betas"])
    return TrainConfig(**raw)


def learning_rate(config: TrainConfig, update: int) -> float:
    """Learning rate of update ``update`` (1-based)."""
    peak = config.peak_learning_rate
    if update <= config.warmup_updates:
        return peak * update / config.warmup_updates
    if update <= config.decay_start:
        return peak
    return peak * (config.total_updates - update) / (config.total_updates - config.decay_start)


def ema_momentum(config: TrainConfig, update: int) -> float:
    """Teacher momentum of update ``update``: constant, then a cosine rise to 1 during the decay."""
    if update <= config.decay_start:
        return config.ema_momentum
    progress = (update - config.decay_start) / (config.total_updates - config.decay_start)
    return 1.0 - (1.0 - config.ema_momentum) * 0.5 * (1.0 + math.cos(math.pi * progress))


class Celia(torch.nn.Module):
    """Student encoder, its two training heads and the EMA teacher."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = CatalogEncoder(config)
        self.property_decoder = PropertyDecoder(config.dim)
        self.teacher_predictor = TeacherPredictor(config.dim, config.heads)
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False).eval()

    def trainable(self) -> list[torch.nn.Module]:
        return [self.encoder, self.property_decoder, self.teacher_predictor]

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for teacher, student in zip(self.teacher.parameters(), self.encoder.parameters()):
            teacher.mul_(momentum).add_(student, alpha=1.0 - momentum)


def make_optimizer(model: Celia, config: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only (biases, norms and the base token are not decayed)."""
    decay, no_decay = [], []
    for module in model.trainable():
        for p in module.parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=config.peak_learning_rate, betas=config.betas)


def training_step(model: Celia, batch, mean, std, generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Losses of one batch (``batch`` already on the device)."""
    values, available, valid = augment_symmetries(batch.values, batch.available, batch.valid, batch.padding, generator)
    view = sample_masks(values, available, valid, batch.padding, batch.host_id, generator)
    x = normalize(values, mean, std)
    space = (batch.box_size_mpc_h, batch.particle_grid_side)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        student = model.encoder(x, view.student_available, view.student_valid, view.observed, batch.padding, *space)
        queries = model.encoder.tokenize(x, view.student_available, view.student_valid, view.observed, batch.padding)
        with torch.no_grad():
            unmasked = available & valid & ~batch.padding.unsqueeze(-1)
            teacher = model.teacher(x, available, valid, unmasked, batch.padding, *space)
        predicted = model.teacher_predictor(queries, student.latent_slots, view.predictor_rows)
        o_loss = teacher_prediction_loss(predicted, teacher.object_features, view.predictor_rows)
    with torch.autocast("cuda", enabled=False):
        properties = model.property_decoder(student.object_features.float(), batch.padding)
        m_loss = masked_property_loss(properties, x, values, view.targets)
    return {"loss": m_loss + TEACHER_PREDICTION_WEIGHT * o_loss, "M": m_loss, "O": o_loss}


def save(path: Path, model: Celia, optimizer, step: int, mean, std, config: TrainConfig) -> None:
    state = {
        "step": step,
        "encoder": model.encoder.state_dict(),
        "property_decoder": model.property_decoder.state_dict(),
        "teacher_predictor": model.teacher_predictor.state_dict(),
        "teacher": model.teacher.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "feature_mean": mean.cpu(),
        "feature_std": std.cpu(),
        "config": asdict(config),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(run_dir: Path, config: TrainConfig, stop: int) -> None:
    """Train the run in ``run_dir`` up to update ``stop`` (continuing from its newest checkpoint)."""
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    start_file = run_dir / "last.pt" if (run_dir / "last.pt").exists() else run_dir / "step0.pt"
    state = torch.load(start_file, map_location=device, weights_only=False)

    model = Celia(config.model).to(device)
    for name in ("encoder", "property_decoder", "teacher_predictor", "teacher"):
        getattr(model, name).load_state_dict(state[name])
    optimizer = make_optimizer(model, config)
    if state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    mean, std = state["feature_mean"].to(device), state["feature_std"].to(device)
    step = state["step"]

    train_ids = split_ids(list(range(1000)), config.split_seed)["train"]
    ids = {suite: train_ids for suite in HYDRO_SUITES}
    dataset = PairedCatalogs(Path(config.tables), ids, config.view_cycle, config.seed)
    sampler = SuiteAlternatingSampler(dataset, config.seed)
    updates_per_epoch = len(sampler)
    log = (run_dir / "log.jsonl").open("a")

    while step < stop:
        epoch, cursor = divmod(step, updates_per_epoch)
        dataset.epoch = sampler.epoch = epoch
        sampler.skip = cursor
        # A fresh loader per epoch, so the workers see the new epoch's view seed.
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler, collate_fn=collate_pairs, num_workers=config.num_workers, pin_memory=True
        )
        for batch in loader:
            if step >= stop:
                break
            update = step + 1
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(config, update)
            generator = torch.Generator(device).manual_seed(config.seed * 1_000_003 + update)
            started = time.time()

            losses = training_step(model, batch.to(device), mean, std, generator)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            parameters = [p for module in model.trainable() for p in module.parameters()]
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm)
            optimizer.step()
            momentum = ema_momentum(config, update)
            model.update_teacher(momentum)
            step = update

            record = {k: float(v.detach()) for k, v in losses.items()}
            record.update(step=step, lr=learning_rate(config, update), ema=momentum, grad_norm=float(grad_norm),
                          seconds=round(time.time() - started, 3))
            log.write(json.dumps(record) + "\n")
            if step % SAVE_EVERY == 0 or step == stop:
                log.flush()
                save(run_dir / "last.pt", model, optimizer, step, mean, std, config)
        if step == stop:
            save(run_dir / f"step{step}.pt", model, optimizer, step, mean, std, config)
    log.close()
