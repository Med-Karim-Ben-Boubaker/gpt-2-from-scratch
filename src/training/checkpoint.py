from __future__ import annotations

import json
import secrets
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACTS_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "training_jobs"


@dataclass
class CheckpointMetadata:
    """Metadata for tracking checkpoint state and best model."""

    training_id: str
    best_step: Optional[int] = None
    best_epoch: Optional[int] = None
    best_metric: float = float("inf")
    total_checkpoints_saved: int = 0
    checkpoint_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointMetadata:
        return cls(**data)


def _generate_training_id() -> str:
    """Generate a random 6-character training ID."""
    return secrets.token_hex(3)


class CheckpointManager:
    """Manages model checkpoints during training.

    Saves checkpoints at regular intervals, tracks the best checkpoint
    based on validation loss, and prunes old checkpoints to keep only
    the most recent N.
    """

    def __init__(
        self,
        training_id: Optional[str] = None,
        checkpoint_cadence: int = 1000,
        max_checkpoints: int = 5,
    ) -> None:
        self.training_id = training_id or _generate_training_id()
        self.checkpoint_cadence = checkpoint_cadence
        self.max_checkpoints = max_checkpoints

        self.checkpoint_dir = ARTIFACTS_ROOT / self.training_id / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.checkpoint_dir / "checkpoint_meta.json"
        self.metadata = self._load_or_create_metadata()

        logger.info("CheckpointManager initialized for training_id=%s", self.training_id)
        logger.info("Checkpoint directory: %s", self.checkpoint_dir)
        logger.info("Checkpoint cadence: every %d steps", self.checkpoint_cadence)
        logger.info("Max checkpoints to keep: %d", self.max_checkpoints)

    def _load_or_create_metadata(self) -> CheckpointMetadata:
        """Load existing metadata or create new."""
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded existing checkpoint metadata")
            return CheckpointMetadata.from_dict(data)
        return CheckpointMetadata(training_id=self.training_id)

    def _save_metadata(self) -> None:
        """Persist metadata to disk."""
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

    def should_checkpoint(self, step: int) -> bool:
        """Determine if a checkpoint should be saved at this step."""
        return step > 0 and step % self.checkpoint_cadence == 0

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        validation_loss: float,
        train_loss: Optional[float] = None,
    ) -> Path:
        """Save a checkpoint and update metadata.

        Args:
            model: The model to save.
            optimizer: The optimizer state to save.
            step: Current training step.
            epoch: Current epoch.
            validation_loss: Current validation loss for best-model tracking.
            train_loss: Optional training loss for logging.

        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint_name = f"checkpoint_step_{step:06d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint_data = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "train_loss": train_loss,
        }

        torch.save(checkpoint_data, checkpoint_path)
        logger.info(
            "Saved checkpoint at step %d (epoch %d) with val_loss=%.4f to %s",
            step,
            epoch,
            validation_loss,
            checkpoint_path,
        )

        self.metadata.checkpoint_files.append(checkpoint_name)
        self.metadata.total_checkpoints_saved += 1

        is_best = validation_loss < self.metadata.best_metric
        if is_best:
            self._update_best_checkpoint(step, epoch, validation_loss)

        self._prune_old_checkpoints()
        self._save_metadata()

        return checkpoint_path

    def _update_best_checkpoint(self, step: int, epoch: int, validation_loss: float) -> None:
        """Update best checkpoint tracking and create symlink."""
        self.metadata.best_step = step
        self.metadata.best_epoch = epoch
        self.metadata.best_metric = validation_loss

        best_link = self.checkpoint_dir / "best_checkpoint.pt"
        checkpoint_name = f"checkpoint_step_{step:06d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()

        shutil.copy2(checkpoint_path, best_link)

        logger.info(
            "New best checkpoint at step %d with val_loss=%.4f",
            step,
            validation_loss,
        )

    def _prune_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent N."""
        while len(self.metadata.checkpoint_files) > self.max_checkpoints:
            oldest_name = self.metadata.checkpoint_files.pop(0)
            oldest_path = self.checkpoint_dir / oldest_name

            if oldest_path.exists():
                best_step = self.metadata.best_step
                is_best = best_step is not None and f"step_{best_step:06d}" in oldest_name

                if not is_best:
                    oldest_path.unlink()
                    logger.debug("Pruned old checkpoint: %s", oldest_name)
                else:
                    logger.debug("Kept best checkpoint despite pruning: %s", oldest_name)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[Path] = None,
        load_best: bool = False,
    ) -> dict:
        """Load a checkpoint into model and optionally optimizer.

        Args:
            model: Model to load weights into.
            optimizer: Optional optimizer to load state into.
            checkpoint_path: Specific checkpoint to load. If None, loads latest or best.
            load_best: If True and checkpoint_path is None, load best checkpoint.

        Returns:
            Dictionary with checkpoint metadata (step, epoch, losses).
        """
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"
            else:
                if not self.metadata.checkpoint_files:
                    raise FileNotFoundError("No checkpoints available to load")
                latest_name = self.metadata.checkpoint_files[-1]
                checkpoint_path = self.checkpoint_dir / latest_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint_data["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        logger.info(
            "Loaded checkpoint from %s (step=%d, epoch=%d, val_loss=%.4f)",
            checkpoint_path,
            checkpoint_data["step"],
            checkpoint_data["epoch"],
            checkpoint_data["validation_loss"],
        )

        return {
            "step": checkpoint_data["step"],
            "epoch": checkpoint_data["epoch"],
            "validation_loss": checkpoint_data["validation_loss"],
            "train_loss": checkpoint_data.get("train_loss"),
        }

    def get_best_checkpoint_info(self) -> Optional[dict]:
        """Get information about the best checkpoint."""
        if self.metadata.best_step is None:
            return None
        return {
            "step": self.metadata.best_step,
            "epoch": self.metadata.best_epoch,
            "validation_loss": self.metadata.best_metric,
            "path": self.checkpoint_dir / "best_checkpoint.pt",
        }

