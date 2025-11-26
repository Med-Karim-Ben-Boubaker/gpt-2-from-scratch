from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.gpt_config import TrainConfig
from src.training.checkpoint import CheckpointManager
from src.training.loss import calc_loss_batch, calc_loss_loader
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Classes for State Management
# =============================================================================


@dataclass
class StepHistory:
    """Tracks training and validation metrics across optimizer steps."""

    training_losses: List[float] = field(default_factory=list)
    validation_losses: List[float] = field(default_factory=list)
    step_numbers: List[int] = field(default_factory=list)
    last_validation_loss: float = float("inf")

    def record_training_step(self, step: int, loss: float) -> None:
        """Record a training loss for an optimizer step (validation TBD)."""
        self.training_losses.append(loss)
        self.validation_losses.append(float("nan"))
        self.step_numbers.append(step)

    def record_validation(self, loss: float) -> None:
        """Update the most recent step with its validation loss."""
        if self.validation_losses:
            self.validation_losses[-1] = loss
        self.last_validation_loss = loss

    def has_validation_at_current_step(self, step: int) -> bool:
        """Check if validation already exists for the given step."""
        if not self.step_numbers or not self.validation_losses:
            return False
        return (
            self.step_numbers[-1] == step
            and not math.isnan(self.validation_losses[-1])
        )


@dataclass
class TrainingContext:
    """Encapsulates all training state and utilities."""

    model: torch.nn.Module
    optimizer: Optimizer
    scheduler: Optional[SequentialLR | LinearLR | CosineAnnealingLR]
    scaler: GradScaler
    checkpoint_manager: CheckpointManager
    device: torch.device
    config: TrainConfig
    writer: Optional[SummaryWriter]

    use_amp: bool = field(init=False)
    current_epoch: int = 0
    optimizer_step: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        self.use_amp = self.config.amp and self.device.type == "cuda"


# =============================================================================
# Factory Functions
# =============================================================================


def _create_optimizer(model: torch.nn.Module, config: TrainConfig) -> Optimizer:
    """Create AdamW optimizer with optional fused implementation."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    use_fused = (
        config.fused
        and torch.cuda.is_available()
        and hasattr(torch.optim.AdamW, "fused")
    )
    return torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=use_fused,
    )


def _create_scheduler(
    optimizer: Optimizer,
    steps_per_epoch: int,
    config: TrainConfig,
) -> SequentialLR | LinearLR | CosineAnnealingLR | None:
    """Create learning rate scheduler with optional warmup and cosine decay."""
    if config.warmup_steps <= 0:
        return None

    total_steps = max(1, steps_per_epoch * config.num_epochs)
    warmup_steps = min(config.warmup_steps, total_steps)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    remaining_steps = max(0, total_steps - warmup_steps)
    if config.min_lr > 0 and remaining_steps > 0:
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=remaining_steps,
            eta_min=config.min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    return warmup_scheduler


def _create_training_context(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    writer: Optional[SummaryWriter],
) -> TrainingContext:
    """Initialize all training components."""
    optimizer = _create_optimizer(model, config)
    steps_per_epoch = max(1, len(train_loader) // max(1, config.grad_accum_steps))
    scheduler = _create_scheduler(optimizer, steps_per_epoch, config)

    checkpoint_manager = CheckpointManager(
        training_id=config.training_id,
        checkpoint_cadence=config.checkpoint_cadence,
        max_checkpoints=config.max_checkpoints,
    )

    is_cuda = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=config.amp and is_cuda)

    return TrainingContext(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        checkpoint_manager=checkpoint_manager,
        device=device,
        config=config,
        writer=writer,
    )


# =============================================================================
# Training Step Helpers
# =============================================================================


def _count_tokens(input_batch: torch.Tensor | List[torch.Tensor]) -> int:
    """Count total tokens in a batch."""
    if isinstance(input_batch, torch.Tensor):
        return int(input_batch.numel())
    return int(sum(tensor.numel() for tensor in input_batch))


def _clip_gradients(ctx: TrainingContext) -> None:
    """Apply gradient clipping if configured."""
    if ctx.config.grad_clip_norm > 0:
        ctx.scaler.unscale_(ctx.optimizer)
        torch.nn.utils.clip_grad_norm_(
            ctx.model.parameters(),
            max_norm=ctx.config.grad_clip_norm,
        )


def _step_optimizer(ctx: TrainingContext) -> None:
    """Execute optimizer step with gradient scaling."""
    ctx.scaler.step(ctx.optimizer)
    ctx.scaler.update()
    ctx.optimizer.zero_grad(set_to_none=True)
    ctx.optimizer_step += 1

    if ctx.scheduler is not None:
        ctx.scheduler.step()


def _get_current_lr(ctx: TrainingContext) -> float:
    """Get current learning rate from scheduler or optimizer."""
    if ctx.scheduler is not None:
        return ctx.scheduler.get_last_lr()[0]
    return ctx.optimizer.param_groups[0]["lr"]


def _log_training_step(ctx: TrainingContext, loss: float) -> None:
    """Log training metrics to TensorBoard."""
    if ctx.writer is None:
        return
    ctx.writer.add_scalar("Loss/Train", loss, ctx.optimizer_step)
    ctx.writer.add_scalar("Learning_Rate", _get_current_lr(ctx), ctx.optimizer_step)


def _run_validation(
    ctx: TrainingContext,
    validation_loader: DataLoader,
    history: StepHistory,
) -> float:
    """Execute validation and record results."""
    ctx.model.eval()
    with torch.no_grad():
        val_loss = calc_loss_loader(
            validation_loader,
            ctx.model,
            ctx.device,
            num_batches=ctx.config.eval_iter,
        )
    ctx.model.train()

    history.record_validation(val_loss)

    if ctx.writer is not None:
        ctx.writer.add_scalar("Loss/Validation", val_loss, ctx.optimizer_step)

    return val_loss


def _maybe_checkpoint(ctx: TrainingContext, train_loss: float, val_loss: float) -> None:
    """Save checkpoint if cadence condition is met."""
    if ctx.checkpoint_manager.should_checkpoint(ctx.optimizer_step):
        ctx.checkpoint_manager.save_checkpoint(
            model=ctx.model,
            optimizer=ctx.optimizer,
            step=ctx.optimizer_step,
            epoch=ctx.current_epoch,
            validation_loss=val_loss,
            train_loss=train_loss,
        )


def _should_validate(ctx: TrainingContext) -> bool:
    """Determine if validation should run at current step."""
    return (
        ctx.optimizer_step > 0
        and ctx.config.eval_freq > 0
        and ctx.optimizer_step % ctx.config.eval_freq == 0
    )


def _flush_gradient_step(
    ctx: TrainingContext,
    validation_loader: DataLoader,
    history: StepHistory,
    accumulated_loss: float,
    micro_batch_count: int,
) -> float:
    """Complete an optimizer step: clip, step, log, validate, checkpoint.

    Returns the average loss for this accumulation window.
    """
    _clip_gradients(ctx)
    _step_optimizer(ctx)

    avg_loss = accumulated_loss / max(1, micro_batch_count)
    history.record_training_step(ctx.optimizer_step, avg_loss)
    _log_training_step(ctx, avg_loss)

    if _should_validate(ctx):
        val_loss = _run_validation(ctx, validation_loader, history)
        logger.info(
            "Step %d: train_loss=%.4f, val_loss=%.4f",
            ctx.optimizer_step,
            avg_loss,
            val_loss,
        )
    else:
        val_loss = history.last_validation_loss

    _maybe_checkpoint(ctx, avg_loss, val_loss)

    return avg_loss


# =============================================================================
# Epoch Processing
# =============================================================================


def _process_epoch(
    ctx: TrainingContext,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    history: StepHistory,
    epoch_index: int,
) -> float:
    """Process a single training epoch.

    Returns the average training loss for the epoch.
    """
    ctx.current_epoch = epoch_index
    epoch_loss_sum = 0.0
    epoch_batch_count = 0
    window_loss_sum = 0.0
    window_micro_batches = 0

    batch_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch_index + 1}/{ctx.config.num_epochs}",
        position=1,
        leave=False,
        total=len(train_loader),
    )

    for batch_idx, (input_batch, target_batch) in enumerate(batch_progress):
        batch_loss = _process_micro_batch(ctx, input_batch, target_batch)

        ctx.total_tokens += _count_tokens(input_batch)
        epoch_loss_sum += batch_loss
        epoch_batch_count += 1
        window_loss_sum += batch_loss
        window_micro_batches += 1

        accumulation_complete = (batch_idx + 1) % ctx.config.grad_accum_steps == 0

        if accumulation_complete:
            avg_loss = _flush_gradient_step(
                ctx, validation_loader, history, window_loss_sum, window_micro_batches
            )
            _update_batch_progress(ctx, batch_progress, avg_loss, history)
            window_loss_sum = 0.0
            window_micro_batches = 0

    # Handle leftover micro-batches at epoch end
    if window_micro_batches > 0:
        avg_loss = _flush_gradient_step(
            ctx, validation_loader, history, window_loss_sum, window_micro_batches
        )
        _update_batch_progress(ctx, batch_progress, avg_loss, history)

    batch_progress.close()
    return epoch_loss_sum / max(1, epoch_batch_count)


def _process_micro_batch(
    ctx: TrainingContext,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
) -> float:
    """Process a single micro-batch: forward pass and scaled backward."""
    with autocast(device_type=ctx.device.type, enabled=ctx.use_amp):
        loss = calc_loss_batch(input_batch, target_batch, ctx.model, ctx.device)
        normalized_loss = loss / ctx.config.grad_accum_steps

    ctx.scaler.scale(normalized_loss).backward()
    return loss.item()


def _update_batch_progress(
    ctx: TrainingContext,
    progress: tqdm,
    train_loss: float,
    history: StepHistory,
) -> None:
    """Update progress bar with current metrics."""
    postfix = {
        "train_loss": f"{train_loss:.4f}",
        "step": ctx.optimizer_step,
        "tokens": f"{ctx.total_tokens:,}",
    }

    if not math.isnan(history.validation_losses[-1]) if history.validation_losses else False:
        postfix["val_loss"] = f"{history.validation_losses[-1]:.4f}"

    progress.set_postfix(postfix)


def _run_end_of_epoch_validation(
    ctx: TrainingContext,
    validation_loader: DataLoader,
    history: StepHistory,
    avg_epoch_loss: float,
) -> None:
    """Run validation at epoch end if not already done at current step."""
    already_validated = history.has_validation_at_current_step(ctx.optimizer_step)

    if already_validated:
        return

    val_loss = _run_validation(ctx, validation_loader, history)

    # Edge case: no optimizer steps yet in this epoch
    if not history.step_numbers:
        history.record_training_step(ctx.optimizer_step, avg_epoch_loss)
        history.record_validation(val_loss)

    logger.info(
        "End-of-epoch %d (step %d): val_loss=%.4f",
        ctx.current_epoch + 1,
        ctx.optimizer_step,
        val_loss,
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def pretrain(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    training_config: TrainConfig,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[int, int, List[float], List[float], List[int], Optimizer, CheckpointManager]:
    """Train a model using standard language model pretraining.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        validation_loader: DataLoader for validation data.
        device: Device to train on.
        training_config: Training hyperparameters and settings.
        writer: Optional TensorBoard writer for logging.

    Returns:
        Tuple containing:
            - Final optimizer step count
            - Total tokens processed
            - List of training losses per step
            - List of validation losses per step (NaN where not computed)
            - List of step numbers
            - The optimizer
            - The checkpoint manager
    """
    model.to(device).train()

    ctx = _create_training_context(model, train_loader, device, training_config, writer)
    history = StepHistory()

    logger.info("Starting pretraining")
    logger.info("Gradient accumulation steps: %s", training_config.grad_accum_steps)
    logger.info("Mixed precision enabled: %s", ctx.use_amp)

    epoch_progress = tqdm(
        range(training_config.num_epochs),
        desc="Epochs",
        position=0,
        leave=True,
    )

    for epoch_index in epoch_progress:
        avg_epoch_loss = _process_epoch(
            ctx, train_loader, validation_loader, history, epoch_index
        )

        _run_end_of_epoch_validation(ctx, validation_loader, history, avg_epoch_loss)

        epoch_progress.set_postfix({
            "avg_train_loss": f"{avg_epoch_loss:.4f}",
            "val_loss": f"{history.last_validation_loss:.4f}",
            "total_steps": ctx.optimizer_step,
        })

    epoch_progress.close()
    logger.info("Pretraining finished")

    best_info = ctx.checkpoint_manager.get_best_checkpoint_info()
    if best_info:
        logger.info(
            "Best checkpoint: step=%d, epoch=%d, val_loss=%.4f",
            best_info["step"],
            best_info["epoch"],
            best_info["validation_loss"],
        )

    return (
        ctx.optimizer_step,
        ctx.total_tokens,
        history.training_losses,
        history.validation_losses,
        history.step_numbers,
        ctx.optimizer,
        ctx.checkpoint_manager,
    )
