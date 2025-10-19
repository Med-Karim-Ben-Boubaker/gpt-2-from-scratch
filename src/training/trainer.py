import os
from typing import Callable, Optional, Tuple, List

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.config import TrainConfig
from src.training.evaluate import calc_loss_batch, calc_loss_loader, calc_masked_loss_batch, calc_masked_loss_loader
from src.training.generate import generate
from src.data.tokenizer import text_to_token_ids, token_ids_to_text
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_eos_token_id(tokenizer) -> int:
    """Get the end-of-sequence token ID for different tokenizer types."""
    try:
        # For BPE tokenizers with vocab property
        if hasattr(tokenizer, 'vocab'):
            # Try to find EOS token in vocabulary
            eos_tokens = ["<eot>", "<eos>", "</s>"]
            for eos_token in eos_tokens:
                if eos_token in tokenizer.vocab:
                    return tokenizer.vocab[eos_token]
            # If no EOS token found, return the last token ID
            return max(tokenizer.vocab.values()) if tokenizer.vocab else 0
        # For other tokenizers
        elif hasattr(tokenizer, 'encode'):
            # Try common EOS tokens
            eos_tokens = ["<eot>", "<eos>", "</s>"]
            for eos_token in eos_tokens:
                try:
                    encoded = tokenizer.encode(eos_token)
                    if encoded:
                        return encoded[0]
                except Exception:
                    continue
            # Fallback to a reasonable default
            return 0
        else:
            logger.warning("Unknown tokenizer type, using default EOS token ID 0")
            return 0
    except Exception as e:
        logger.warning(f"Failed to get EOS token ID: {e}, using default 0")
        return 0

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    training_config: TrainConfig,
    tokenizer,
    on_step_callback: Optional[Callable[[int, float, int], None]] = None,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[int, int, List[float], List[float], List[int], Optimizer]:
    is_cuda = device.type == "cuda"
    mixed_precision_scaler = GradScaler("cuda", enabled=training_config.amp and is_cuda)

    total_tokens_processed = 0
    optimizer_update_step = 0

    training_losses = []
    validation_losses = []
    step_numbers = []

    model.to(device).train()

    # Create optimizer and scheduler inline
    params = [param for param in model.parameters() if param.requires_grad]
    
    # Check if fused AdamW is available (requires CUDA and modern GPU)
    use_fused = (
        training_config.fused
        and torch.cuda.is_available()
        and hasattr(torch.optim.AdamW, "fused")
    )
    
    optimizer = torch.optim.AdamW(
        params,
        lr=training_config.lr,
        betas=training_config.betas,
        eps=training_config.eps,
        fused=use_fused,
    )

    # Estimate total steps for scheduler
    total_steps = len(train_loader) // training_config.grad_accum_steps * training_config.num_epochs
    
    # Create scheduler inline
    scheduler = None
    if training_config.warmup_steps > 0:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

        # Linear warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=training_config.warmup_steps,
        )

        # If min_lr is set, add cosine decay after warmup
        if training_config.min_lr > 0:
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - training_config.warmup_steps,
                eta_min=training_config.min_lr,
            )
            scheduler = SequentialLR(
                optimizer,
                [warmup_scheduler, cosine_scheduler],
                milestones=[training_config.warmup_steps],
            )
        else:
            scheduler = warmup_scheduler

    logger.info("Created optimizer: AdamW")
    if scheduler:
        logger.info(
            f"Created scheduler: warmup_steps={training_config.warmup_steps}, total_steps={total_steps}"
        )

    # Set inference prompt for text generation during validation
    inference_prompt = "A cat is sleeping on the sofa, the dog"
    context_length = getattr(model, "pos_emb", None)
    if context_length is not None:
        context_length = context_length.weight.shape[0]
    else:
        context_length = 1024  # Default fallback

    logger.info(f"Starting training for {training_config.num_epochs} epochs...")
    logger.info(f"Gradient accumulation steps: {training_config.grad_accum_steps}")
    logger.info(f"Mixed precision training: {training_config.amp and is_cuda}")

    # Create epoch progress bar
    epoch_pbar = tqdm(
        range(training_config.num_epochs), desc="Epochs", position=0, leave=True
    )

    for current_epoch in epoch_pbar:
        epoch_training_loss = 0.0
        epoch_batches_processed = 0

        # Create batch progress bar for current epoch
        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {current_epoch + 1}/{training_config.num_epochs}",
            position=1,
            leave=False,
            total=len(train_loader),
        )

        # Accumulation window stats for logging per optimizer update
        window_loss_sum = 0.0
        window_micro_batches = 0
        # Throttle progress bar updates to reduce overhead
        postfix_step_freq = max(1, int(os.environ.get("POSTFIX_STEP_FREQ", "10")))
        clip_grad_norm = training_config.grad_clip_norm

        for batch_index, (input_tokens, target_tokens) in enumerate(batch_pbar):
            with autocast("cuda", enabled=training_config.amp and is_cuda):
                batch_loss = calc_loss_batch(input_tokens, target_tokens, model, device)
                scaled_loss = batch_loss / training_config.grad_accum_steps

            mixed_precision_scaler.scale(scaled_loss).backward()

            is_accumulation_step_complete = (
                batch_index + 1
            ) % training_config.grad_accum_steps == 0

            if is_accumulation_step_complete:
                # Gradient clipping (after unscale when using AMP)
                if clip_grad_norm > 0:
                    mixed_precision_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad_norm
                    )
                mixed_precision_scaler.step(optimizer)
                mixed_precision_scaler.update()
                optimizer.zero_grad(set_to_none=True)

                optimizer_update_step += 1

                # PyTorch scheduler step (much more efficient!)
                if scheduler is not None:
                    scheduler.step()

            batch_tokens = sum(seq.numel() for seq in input_tokens)
            total_tokens_processed += batch_tokens

            # Track unscaled loss for correct averaging and epoch statistics
            epoch_training_loss += batch_loss.item()
            window_loss_sum += batch_loss.item()
            window_micro_batches += 1
            epoch_batches_processed += 1

            if is_accumulation_step_complete:
                # Record average loss over the accumulation window per optimizer update
                avg_update_loss = window_loss_sum / max(1, window_micro_batches)
                training_losses.append(avg_update_loss)
                step_numbers.append(optimizer_update_step)

                # Log to TensorBoard
                if writer is not None:
                    writer.add_scalar(
                        "Loss/Train", avg_update_loss, optimizer_update_step
                    )
                    # Log current learning rate
                    current_lr = (
                        scheduler.get_last_lr()[0]
                        if scheduler
                        else optimizer.param_groups[0]["lr"]
                    )
                    writer.add_scalar(
                        "Learning_Rate", current_lr, optimizer_update_step
                    )

                # Reset window stats
                window_loss_sum = 0.0
                window_micro_batches = 0

            if on_step_callback:
                on_step_callback(
                    optimizer_update_step, scaled_loss.item(), total_tokens_processed
                )

            should_evaluate = (
                is_accumulation_step_complete
                and optimizer_update_step % training_config.eval_freq == 0
                and optimizer_update_step > 0
            )

            if should_evaluate:
                # Update batch progress bar to show evaluation
                batch_pbar.set_postfix(
                    {
                        "train_loss": f"{scaled_loss.item():.4f}",
                        "status": "evaluating...",
                    }
                )

                model.eval()
                with torch.no_grad():
                    current_validation_loss = calc_loss_loader(
                        validation_loader,
                        model,
                        device,
                        num_batches=training_config.eval_iter,
                    )

                    # Generate sample text to monitor training progress
                    try:
                        input_token_ids = text_to_token_ids(
                            inference_prompt, tokenizer
                        ).to(device)
                        eos_token_id = get_eos_token_id(tokenizer)
                        generated_token_ids, _ = generate(
                            model,
                            input_token_ids,
                            max_new_tokens=25,
                            context_size=context_length,
                            temperature=0.8,
                            top_p=0.95,
                            repetition_penalty=1.1,
                            eos_token_id=eos_token_id,
                            min_new_tokens=5,
                        )
                        generated_text = token_ids_to_text(
                            generated_token_ids, tokenizer
                        )
                        logger.info(
                            f"Step {optimizer_update_step} | Generated: {generated_text}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Text generation failed at step {optimizer_update_step}: {e}"
                        )

                model.train()

                validation_losses.append(current_validation_loss)

                # Log validation loss to TensorBoard
                if writer is not None:
                    writer.add_scalar(
                        "Loss/Validation",
                        current_validation_loss,
                        optimizer_update_step,
                    )
                    writer.add_text(
                        "Generated_Text", generated_text, optimizer_update_step
                    )

                # Update progress bars with evaluation results
                batch_pbar.set_postfix(
                    {
                        "train_loss": f"{scaled_loss.item():.4f}",
                        "val_loss": f"{current_validation_loss:.4f}",
                        "step": optimizer_update_step,
                        "tokens": f"{total_tokens_processed:,}",
                    }
                )

                # Update epoch progress bar with latest metrics
                epoch_pbar.set_postfix(
                    {
                        "avg_train_loss": f"{epoch_training_loss / max(1, epoch_batches_processed):.4f}",
                        "val_loss": f"{current_validation_loss:.4f}",
                        "step": optimizer_update_step,
                    }
                )
            elif is_accumulation_step_complete:
                # Add NaN for validation loss when not evaluating but at gradient step
                validation_losses.append(float("nan"))

                # Throttled progress bar updates
                if (optimizer_update_step % postfix_step_freq) == 0:
                    batch_pbar.set_postfix(
                        {
                            "train_loss": f"{(training_losses[-1] if training_losses else scaled_loss.item()):.4f}",
                            "step": optimizer_update_step,
                            "tokens": f"{total_tokens_processed:,}",
                        }
                    )

        # Flush leftover gradients if the last accumulation window is incomplete
        if window_micro_batches > 0:
            if clip_grad_norm > 0:
                mixed_precision_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_grad_norm
                )
            mixed_precision_scaler.step(optimizer)
            mixed_precision_scaler.update()
            optimizer.zero_grad(set_to_none=True)

            optimizer_update_step += 1

            # PyTorch scheduler step when flushing at epoch end
            if scheduler is not None:
                scheduler.step()

            avg_update_loss = window_loss_sum / max(1, window_micro_batches)
            training_losses.append(avg_update_loss)
            step_numbers.append(optimizer_update_step)
            # For alignment, append NaN for val loss unless we evaluate below
            validation_losses.append(float("nan"))

            # Optional evaluation if step aligns with eval frequency
            if (optimizer_update_step % training_config.eval_freq) == 0:
                model.eval()
                with torch.no_grad():
                    current_validation_loss = calc_loss_loader(
                        validation_loader,
                        model,
                        device,
                        num_batches=training_config.eval_iter,
                    )

                    # Generate sample text to monitor training progress
                    try:
                        input_token_ids = text_to_token_ids(
                            inference_prompt, tokenizer
                        ).to(device)
                        eos_token_id = get_eos_token_id(tokenizer)
                        generated_token_ids, _ = generate(
                            model,
                            input_token_ids,
                            max_new_tokens=25,
                            context_size=context_length,
                            temperature=0.8,
                            top_p=0.95,
                            repetition_penalty=1.1,
                            eos_token_id=eos_token_id,
                            min_new_tokens=5,
                        )
                        generated_text = token_ids_to_text(
                            generated_token_ids, tokenizer
                        )
                        logger.info(
                            f"Step {optimizer_update_step} (end flush) | Generated: {generated_text}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Text generation failed at step {optimizer_update_step}: {e}"
                        )

                model.train()
                validation_losses[-1] = current_validation_loss

        average_epoch_loss = epoch_training_loss / max(1, epoch_batches_processed)

        model.eval()
        with torch.no_grad():
            final_validation_loss = calc_loss_loader(
                validation_loader, model, device, num_batches=training_config.eval_iter
            )

            # Generate sample text at end of epoch
            try:
                input_token_ids = text_to_token_ids(inference_prompt, tokenizer).to(
                    device
                )
                eos_token_id = get_eos_token_id(tokenizer)
                generated_token_ids, _ = generate(
                    model,
                    input_token_ids,
                    max_new_tokens=25,
                    context_size=context_length,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    eos_token_id=eos_token_id,
                    min_new_tokens=5,
                )
                generated_text = token_ids_to_text(generated_token_ids, tokenizer)
                logger.info(
                    f"End of Epoch {current_epoch + 1} | Generated: {generated_text}"
                )
            except Exception as e:
                logger.warning(
                    f"Text generation failed at end of epoch {current_epoch + 1}: {e}"
                )

        model.train()

        # Note: Epoch-level metrics removed to keep TensorBoard clean

        # Update epoch progress bar with final epoch metrics
        epoch_pbar.set_postfix(
            {
                "avg_train_loss": f"{average_epoch_loss:.4f}",
                "final_val_loss": f"{final_validation_loss:.4f}",
                "total_steps": optimizer_update_step,
            }
        )

        # Close batch progress bar for this epoch
        batch_pbar.close()

    # Close epoch progress bar
    epoch_pbar.close()

    logger.info("Training completed!")
    return (
        optimizer_update_step,
        total_tokens_processed,
        training_losses,
        validation_losses,
        step_numbers,
        optimizer,
    )


def train_instruction_finetuning(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    training_config: TrainConfig,
    tokenizer,
    on_step_callback: Optional[Callable[[int, float, int], None]] = None,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[int, int, List[float], List[float], List[int], Optimizer]:
    """
    Train model for instruction fine-tuning with masked loss.
    """
    is_cuda = device.type == "cuda"
    if device.type == "cpu":
        logger.warning("CUDA is not available. Training will be slow.")
        raise RuntimeError("CUDA is not available. Please check your GPU configuration.")
    
    mixed_precision_scaler = GradScaler("cuda", enabled=training_config.amp and is_cuda)

    total_tokens_processed = 0
    optimizer_update_step = 0

    training_losses = []
    validation_losses = []
    step_numbers = []

    model.to(device).train()

    # Create optimizer and scheduler inline
    params = [param for param in model.parameters() if param.requires_grad]
    
    # Check if fused AdamW is available (requires CUDA and modern GPU)
    use_fused = (
        training_config.fused
        and torch.cuda.is_available()
        and hasattr(torch.optim.AdamW, "fused")
    )
    
    optimizer = torch.optim.AdamW(
        params,
        lr=training_config.lr,
        betas=training_config.betas,
        eps=training_config.eps,
        fused=use_fused,
    )

    # Estimate total steps for scheduler
    total_steps = len(train_loader) // training_config.grad_accum_steps * training_config.num_epochs
    
    # Create scheduler inline
    scheduler = None
    if training_config.warmup_steps > 0:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

        # Linear warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=training_config.warmup_steps,
        )

        # If min_lr is set, add cosine decay after warmup
        if training_config.min_lr > 0:
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - training_config.warmup_steps,
                eta_min=training_config.min_lr,
            )
            scheduler = SequentialLR(
                optimizer,
                [warmup_scheduler, cosine_scheduler],
                milestones=[training_config.warmup_steps],
            )
        else:
            scheduler = warmup_scheduler

    logger.info("Created optimizer: AdamW for instruction fine-tuning")
    if scheduler:
        logger.info(
            f"Created scheduler: warmup_steps={training_config.warmup_steps}, total_steps={total_steps}"
        )

    # Set inference prompt for text generation during validation
    inference_prompt = "Instruction:\nWhat is the capital of France?\n\nResponse:\n"
    context_length = getattr(model, "pos_emb", None)
    if context_length is not None:
        context_length = context_length.weight.shape[0]
    else:
        context_length = 256  # Default fallback

    logger.info(f"Starting instruction fine-tuning for {training_config.num_epochs} epochs...")
    logger.info(f"Gradient accumulation steps: {training_config.grad_accum_steps}")
    logger.info(f"Mixed precision training: {training_config.amp and is_cuda}")

    # Create epoch progress bar
    epoch_pbar = tqdm(
        range(training_config.num_epochs), desc="Epochs", position=0, leave=True
    )

    for current_epoch in epoch_pbar:
        epoch_training_loss = 0.0
        epoch_batches_processed = 0

        # Create batch progress bar for current epoch
        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {current_epoch + 1}/{training_config.num_epochs}",
            position=1,
            leave=False,
            total=len(train_loader),
        )

        # Accumulation window stats for logging per optimizer update
        window_loss_sum = 0.0
        window_micro_batches = 0
        # Throttle progress bar updates to reduce overhead
        postfix_step_freq = max(1, int(os.environ.get("POSTFIX_STEP_FREQ", "10")))
        clip_grad_norm = training_config.grad_clip_norm

        for batch_index, (input_list, target_list, loss_mask_list) in enumerate(batch_pbar):
            with autocast("cuda", enabled=training_config.amp and is_cuda):
                batch_loss = calc_masked_loss_batch(input_list, target_list, loss_mask_list, model, device)
                scaled_loss = batch_loss / training_config.grad_accum_steps

            mixed_precision_scaler.scale(scaled_loss).backward()

            is_accumulation_step_complete = (
                batch_index + 1
            ) % training_config.grad_accum_steps == 0

            if is_accumulation_step_complete:
                # Gradient clipping (after unscale when using AMP)
                if clip_grad_norm > 0:
                    mixed_precision_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad_norm
                    )
                mixed_precision_scaler.step(optimizer)
                mixed_precision_scaler.update()
                optimizer.zero_grad(set_to_none=True)

                optimizer_update_step += 1

                # PyTorch scheduler step (much more efficient!)
                if scheduler is not None:
                    scheduler.step()

            batch_tokens = sum(seq.numel() for seq in input_list)
            total_tokens_processed += batch_tokens

            # Track unscaled loss for correct averaging and epoch statistics
            epoch_training_loss += batch_loss.item()
            window_loss_sum += batch_loss.item()
            window_micro_batches += 1
            epoch_batches_processed += 1

            if is_accumulation_step_complete:
                # Record average loss over the accumulation window per optimizer update
                avg_update_loss = window_loss_sum / max(1, window_micro_batches)
                training_losses.append(avg_update_loss)
                step_numbers.append(optimizer_update_step)

                # Log to TensorBoard
                if writer is not None:
                    writer.add_scalar(
                        "Loss/Train", avg_update_loss, optimizer_update_step
                    )
                    # Log current learning rate
                    current_lr = (
                        scheduler.get_last_lr()[0]
                        if scheduler
                        else optimizer.param_groups[0]["lr"]
                    )
                    writer.add_scalar(
                        "Learning_Rate", current_lr, optimizer_update_step
                    )

                # Reset window stats
                window_loss_sum = 0.0
                window_micro_batches = 0

            if on_step_callback:
                on_step_callback(
                    optimizer_update_step, scaled_loss.item(), total_tokens_processed
                )

            should_evaluate = (
                is_accumulation_step_complete
                and optimizer_update_step % training_config.eval_freq == 0
                and optimizer_update_step > 0
            )

            if should_evaluate:
                # Update batch progress bar to show evaluation
                batch_pbar.set_postfix(
                    {
                        "train_loss": f"{scaled_loss.item():.4f}",
                        "status": "evaluating...",
                    }
                )

                model.eval()
                with torch.no_grad():
                    current_validation_loss = calc_masked_loss_loader(
                        validation_loader,
                        model,
                        device,
                        num_batches=training_config.eval_iter,
                    )

                    # Generate sample text to monitor training progress
                    try:
                        input_token_ids = text_to_token_ids(
                            inference_prompt, tokenizer
                        ).to(device)
                        eos_token_id = get_eos_token_id(tokenizer)
                        generated_token_ids, _ = generate(
                            model,
                            input_token_ids,
                            max_new_tokens=25,
                            context_size=context_length,
                            temperature=0.8,
                            top_p=0.95,
                            repetition_penalty=1.1,
                            eos_token_id=eos_token_id,
                            min_new_tokens=5,
                        )
                        generated_text = token_ids_to_text(
                            generated_token_ids, tokenizer
                        )
                        logger.info(
                            f"Step {optimizer_update_step} | Generated: {generated_text}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Text generation failed at step {optimizer_update_step}: {e}"
                        )

                model.train()

                validation_losses.append(current_validation_loss)

                # Log validation loss to TensorBoard
                if writer is not None:
                    writer.add_scalar(
                        "Loss/Validation",
                        current_validation_loss,
                        optimizer_update_step,
                    )
                    writer.add_text(
                        "Generated_Text", generated_text, optimizer_update_step
                    )

                # Update progress bars with evaluation results
                batch_pbar.set_postfix(
                    {
                        "train_loss": f"{scaled_loss.item():.4f}",
                        "val_loss": f"{current_validation_loss:.4f}",
                        "step": optimizer_update_step,
                        "tokens": f"{total_tokens_processed:,}",
                    }
                )

                # Update epoch progress bar with latest metrics
                epoch_pbar.set_postfix(
                    {
                        "avg_train_loss": f"{epoch_training_loss / max(1, epoch_batches_processed):.4f}",
                        "val_loss": f"{current_validation_loss:.4f}",
                        "step": optimizer_update_step,
                    }
                )
            elif is_accumulation_step_complete:
                # Add NaN for validation loss when not evaluating but at gradient step
                validation_losses.append(float("nan"))

                # Throttled progress bar updates
                if (optimizer_update_step % postfix_step_freq) == 0:
                    batch_pbar.set_postfix(
                        {
                            "train_loss": f"{(training_losses[-1] if training_losses else scaled_loss.item()):.4f}",
                            "step": optimizer_update_step,
                            "tokens": f"{total_tokens_processed:,}",
                        }
                    )

        # Flush leftover gradients if the last accumulation window is incomplete
        if window_micro_batches > 0:
            if clip_grad_norm > 0:
                mixed_precision_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_grad_norm
                )
            mixed_precision_scaler.step(optimizer)
            mixed_precision_scaler.update()
            optimizer.zero_grad(set_to_none=True)

            optimizer_update_step += 1

            # PyTorch scheduler step when flushing at epoch end
            if scheduler is not None:
                scheduler.step()

            avg_update_loss = window_loss_sum / max(1, window_micro_batches)
            training_losses.append(avg_update_loss)
            step_numbers.append(optimizer_update_step)
            # For alignment, append NaN for val loss unless we evaluate below
            validation_losses.append(float("nan"))

            # Optional evaluation if step aligns with eval frequency
            if (optimizer_update_step % training_config.eval_freq) == 0:
                model.eval()
                with torch.no_grad():
                    current_validation_loss = calc_masked_loss_loader(
                        validation_loader,
                        model,
                        device,
                        num_batches=training_config.eval_iter,
                    )

                    # Generate sample text to monitor training progress
                    try:
                        input_token_ids = text_to_token_ids(
                            inference_prompt, tokenizer
                        ).to(device)
                        eos_token_id = get_eos_token_id(tokenizer)
                        generated_token_ids, _ = generate(
                            model,
                            input_token_ids,
                            max_new_tokens=25,
                            context_size=context_length,
                            temperature=0.8,
                            top_p=0.95,
                            repetition_penalty=1.1,
                            eos_token_id=eos_token_id,
                            min_new_tokens=5,
                        )
                        generated_text = token_ids_to_text(
                            generated_token_ids, tokenizer
                        )
                        logger.info(
                            f"Step {optimizer_update_step} (end flush) | Generated: {generated_text}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Text generation failed at step {optimizer_update_step}: {e}"
                        )

                model.train()
                validation_losses[-1] = current_validation_loss

        average_epoch_loss = epoch_training_loss / max(1, epoch_batches_processed)

        model.eval()
        with torch.no_grad():
            final_validation_loss = calc_masked_loss_loader(
                validation_loader, model, device, num_batches=training_config.eval_iter
            )

            # Generate sample text at end of epoch
            try:
                input_token_ids = text_to_token_ids(inference_prompt, tokenizer).to(
                    device
                )
                eos_token_id = get_eos_token_id(tokenizer)
                generated_token_ids, _ = generate(
                    model,
                    input_token_ids,
                    max_new_tokens=25,
                    context_size=context_length,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    eos_token_id=eos_token_id,
                    min_new_tokens=5,
                )
                generated_text = token_ids_to_text(generated_token_ids, tokenizer)
                logger.info(
                    f"End of Epoch {current_epoch + 1} | Generated: {generated_text}"
                )
            except Exception as e:
                logger.warning(
                    f"Text generation failed at end of epoch {current_epoch + 1}: {e}"
                )

        model.train()

        # Note: Epoch-level metrics removed to keep TensorBoard clean

        # Update epoch progress bar with final epoch metrics
        epoch_pbar.set_postfix(
            {
                "avg_train_loss": f"{average_epoch_loss:.4f}",
                "final_val_loss": f"{final_validation_loss:.4f}",
                "total_steps": optimizer_update_step,
            }
        )

        # Close batch progress bar for this epoch
        batch_pbar.close()

    # Close epoch progress bar
    epoch_pbar.close()

    logger.info("Instruction fine-tuning completed!")
    return (
        optimizer_update_step,
        total_tokens_processed,
        training_losses,
        validation_losses,
        step_numbers,
        optimizer,
    )
