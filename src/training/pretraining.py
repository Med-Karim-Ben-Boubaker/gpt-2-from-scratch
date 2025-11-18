from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.gpt_config import TrainConfig
from src.training.loss import calc_loss_batch, calc_loss_loader
from src.utils.logging import get_logger


logger = get_logger(__name__)


def _count_tokens(input_batch: torch.Tensor | List[torch.Tensor]) -> int:
    if isinstance(input_batch, torch.Tensor):
        return int(input_batch.numel())
    return int(sum(tensor.numel() for tensor in input_batch))


def _create_optimizer(model: torch.nn.Module, training_config: TrainConfig) -> Optimizer:
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    use_fused = (
        training_config.fused
        and torch.cuda.is_available()
        and hasattr(torch.optim.AdamW, "fused")
    )
    return torch.optim.AdamW(
        trainable_params,
        lr=training_config.lr,
        betas=training_config.betas,
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
        fused=use_fused,
    )


def _create_scheduler(
    optimizer: Optimizer,
    steps_per_epoch: int,
    training_config: TrainConfig,
) -> SequentialLR | LinearLR | CosineAnnealingLR | None:
    if training_config.warmup_steps <= 0:
        return None

    total_steps = max(1, steps_per_epoch * training_config.num_epochs)
    warmup_steps = min(training_config.warmup_steps, total_steps)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    remaining_steps = max(0, total_steps - warmup_steps)
    if training_config.min_lr > 0 and remaining_steps > 0:
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=remaining_steps,
            eta_min=training_config.min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    return warmup_scheduler


def pretrain(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    training_config: TrainConfig,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[int, int, List[float], List[float], List[int], Optimizer]:
    model.to(device).train()

    total_tokens_processed = 0
    optimizer_update_step = 0
    training_losses: List[float] = []
    validation_losses: List[float] = []
    step_numbers: List[int] = []

    is_cuda = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=training_config.amp and is_cuda)

    optimizer = _create_optimizer(model, training_config)
    steps_per_epoch = max(1, len(train_loader) // max(1, training_config.grad_accum_steps))
    scheduler = _create_scheduler(optimizer, steps_per_epoch, training_config)

    logger.info("Starting pretraining")
    logger.info("Gradient accumulation steps: %s", training_config.grad_accum_steps)
    logger.info("Mixed precision enabled: %s", training_config.amp and is_cuda)

    epoch_progress = tqdm(range(training_config.num_epochs), desc="Epochs", position=0, leave=True)

    for epoch_index in epoch_progress:
        epoch_loss_total = 0.0
        epoch_batches = 0
        window_loss_total = 0.0
        window_micro_batches = 0

        batch_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch_index + 1}/{training_config.num_epochs}",
            position=1,
            leave=False,
            total=len(train_loader),
        )

        for batch_idx, (input_batch, target_batch) in enumerate(batch_progress):
            with autocast(device_type=device.type, enabled=training_config.amp and is_cuda):
                batch_loss = calc_loss_batch(input_batch, target_batch, model, device)
                normalized_loss = batch_loss / training_config.grad_accum_steps

            scaler.scale(normalized_loss).backward()

            accumulation_complete = (batch_idx + 1) % training_config.grad_accum_steps == 0

            if accumulation_complete:
                if training_config.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=training_config.grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                optimizer_update_step += 1
                if scheduler is not None:
                    scheduler.step()

            batch_token_count = _count_tokens(input_batch)
            total_tokens_processed += batch_token_count

            epoch_loss_total += batch_loss.item()
            epoch_batches += 1
            window_loss_total += batch_loss.item()
            window_micro_batches += 1

            if accumulation_complete:
                average_loss = window_loss_total / max(1, window_micro_batches)
                training_losses.append(average_loss)
                validation_losses.append(float("nan"))
                step_numbers.append(optimizer_update_step)
                if writer is not None:
                    writer.add_scalar("Loss/Train", average_loss, optimizer_update_step)
                    current_lr = (
                        scheduler.get_last_lr()[0]
                        if scheduler is not None
                        else optimizer.param_groups[0]["lr"]
                    )
                    writer.add_scalar("Learning_Rate", current_lr, optimizer_update_step)

                batch_progress.set_postfix(
                    {
                        "train_loss": f"{average_loss:.4f}",
                        "step": optimizer_update_step,
                        "tokens": f"{total_tokens_processed:,}",
                    }
                )

                window_loss_total = 0.0
                window_micro_batches = 0

                should_validate = (
                    optimizer_update_step > 0
                    and optimizer_update_step % training_config.eval_freq == 0
                )

                if should_validate:
                    model.eval()
                    with torch.no_grad():
                        validation_loss = calc_loss_loader(
                            validation_loader,
                            model,
                            device,
                            num_batches=training_config.eval_iter,
                        )
                    model.train()

                    validation_losses[-1] = validation_loss
                    logger.info(
                        "Intermediate validation at step %d: train_loss=%.4f, val_loss=%.4f",
                        optimizer_update_step,
                        average_loss,
                        validation_loss,
                    )
                    if writer is not None:
                        writer.add_scalar(
                            "Loss/Validation",
                            validation_loss,
                            optimizer_update_step,
                        )
                    batch_progress.set_postfix(
                        {
                            "train_loss": f"{average_loss:.4f}",
                            "val_loss": f"{validation_loss:.4f}",
                            "step": optimizer_update_step,
                        }
                    )

        # Handles leftover micro-batches at end of epoch that don't complete a full accumulation window
        if window_micro_batches > 0:
            if training_config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            optimizer_update_step += 1
            if scheduler is not None:
                scheduler.step()

            average_loss = window_loss_total / max(1, window_micro_batches)
            training_losses.append(average_loss)
            validation_losses.append(float("nan"))
            step_numbers.append(optimizer_update_step)
            if writer is not None:
                writer.add_scalar("Loss/Train", average_loss, optimizer_update_step)
                current_lr = (
                    scheduler.get_last_lr()[0]
                    if scheduler is not None
                    else optimizer.param_groups[0]["lr"]
                )
                writer.add_scalar("Learning_Rate", current_lr, optimizer_update_step)

            # Check if validation should run after this update
            should_validate = (
                optimizer_update_step > 0
                and optimizer_update_step % training_config.eval_freq == 0
            )

            if should_validate:
                model.eval()
                with torch.no_grad():
                    validation_loss = calc_loss_loader(
                        validation_loader,
                        model,
                        device,
                        num_batches=training_config.eval_iter,
                    )
                model.train()

                validation_losses[-1] = validation_loss
                logger.info(
                    "Validation at step %d: loss=%.4f",
                    optimizer_update_step,
                    validation_loss,
                )
                if writer is not None:
                    writer.add_scalar(
                        "Loss/Validation",
                        validation_loss,
                        optimizer_update_step,
                    )

        average_epoch_loss = epoch_loss_total / max(1, epoch_batches)

        # End-of-epoch validation: only run if we haven't already validated at this step
        last_step_has_validation = (
            step_numbers
            and validation_losses
            and step_numbers[-1] == optimizer_update_step
            and not math.isnan(validation_losses[-1])
        )
        should_validate_end_of_epoch = not last_step_has_validation

        if should_validate_end_of_epoch:
            model.eval()
            with torch.no_grad():
                final_validation_loss = calc_loss_loader(
                    validation_loader,
                    model,
                    device,
                    num_batches=training_config.eval_iter,
                )
            model.train()

            # Update the last validation loss if we have steps recorded
            if step_numbers:
                validation_losses[-1] = final_validation_loss
            else:
                # Edge case: no optimizer steps yet, but we still want to validate
                training_losses.append(average_epoch_loss)
                validation_losses.append(final_validation_loss)
                step_numbers.append(optimizer_update_step)

            logger.info(
                "End-of-epoch validation (epoch %d, step %d): loss=%.4f",
                epoch_index + 1,
                optimizer_update_step,
                final_validation_loss,
            )
            if writer is not None:
                writer.add_scalar(
                    "Loss/Validation",
                    final_validation_loss,
                    optimizer_update_step,
                )

        epoch_progress.set_postfix(
            {
                "avg_train_loss": f"{average_epoch_loss:.4f}",
                "val_loss": f"{validation_losses[-1]:.4f}" if validation_losses else "N/A",
                "total_steps": optimizer_update_step,
            }
        )

        batch_progress.close()

    epoch_progress.close()
    logger.info("Pretraining finished")

    return (
        optimizer_update_step,
        total_tokens_processed,
        training_losses,
        validation_losses,
        step_numbers,
        optimizer,
    )

