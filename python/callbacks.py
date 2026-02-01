"""Self Write Training Callbacks"""

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.weight_averaging import WeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger


class LogWrongGuessesCallback(Callback):
    def __init__(self, classes: list[str], max_images: int = 32) -> None:
        super().__init__()
        self.classes = classes
        self.max_images = max_images
        self.logged_images = 0

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        if self.logged_images >= self.max_images:
            return

        # Check if outputs is available (requires test_step to return pred)
        if outputs is None:
            return

        pred_logits = outputs
        image, label = batch["image"], batch["label"]

        preds = torch.argmax(pred_logits, dim=1)

        # Identify wrong guesses
        mask = preds != label

        if not mask.any():
            return

        # images transformed as float32, converting back
        wrong_images: torch.Tensor = image[mask] * 256
        wrong_images = wrong_images.to(dtype=torch.uint8)
        wrong_preds = preds[mask]
        true_labels = label[mask]

        # Limit the number of images to log
        num_to_log = min(len(wrong_images), self.max_images - self.logged_images)
        wrong_images = wrong_images[:num_to_log]
        wrong_preds = wrong_preds[:num_to_log]
        true_labels = true_labels[:num_to_log]

        # Log to TensorBoard if available
        if isinstance(trainer.logger, TensorBoardLogger):
            # Add text labels
            captions = []
            for p, t in zip(wrong_preds, true_labels):
                pred_name = self.classes[p] if p < len(self.classes) else str(p.item())
                true_name = self.classes[t] if t < len(self.classes) else str(t.item())
                captions.append(f"P: {pred_name} | T: {true_name}")

            tensorboard = trainer.logger.experiment
            tensorboard.add_images(
                "wrong_guesses",
                wrong_images,
                global_step=batch_idx,  # Use batch_idx or a counter
            )

            # Also log text mapping
            tensorboard.add_text(
                "wrong_guesses_text", "  \n".join(captions), global_step=batch_idx
            )

        self.logged_images += num_to_log


def get_ema_multi_avg_fn(
    decay: float = 0.995,
    use_warmup: bool = True,
    min_decay: float = 0.0,
    warmup_gamma: float = 1.0,
    warmup_power: float = 0.7,
):
    """
    Get a multi_avg_fn applying EMA with Inverse Gamma warmup schedule,
    adapted from torch lightning's and timm's implementation

    Unlike the standard get_ema_avg_fn which uses a fixed decay, this version
    calculates the decay dynamically based on the step count (num_averaged).
    This allows you to start EMA immediately (step 0) without initialization bias
    """

    @torch.no_grad()
    def ema_multi_update(
        averaged_param_list: list[torch.Tensor],
        current_param_list: list[torch.Tensor],
        num_averaged: torch.Tensor,
    ):
        step = num_averaged.item()

        # Warmup

        # Formula: decay = 1 - (1 + step / gamma) ^ -power
        if use_warmup:
            cur_decay = 1 - (1 + step / warmup_gamma) ** -warmup_power
            cur_decay = max(min(decay, cur_decay), min_decay)
        else:
            cur_decay = decay

        # Optimization: Filter & Fused Update

        lerp_ema_params = []
        lerp_curr_params = []

        copy_ema_params = []
        copy_curr_params = []

        for ema_p, curr_p in zip(averaged_param_list, current_param_list):
            if ema_p.is_floating_point() or ema_p.is_complex():
                lerp_ema_params.append(ema_p)
                lerp_curr_params.append(curr_p)
            else:
                copy_ema_params.append(ema_p)
                copy_curr_params.append(curr_p)

        # Apply Fused Update (Horizontal Fusion)
        if lerp_ema_params:
            torch._foreach_lerp_(
                lerp_ema_params, lerp_curr_params, weight=1.0 - cur_decay
            )

        # Apply Standard Copy for integers
        for ema_p, curr_p in zip(copy_ema_params, copy_curr_params):
            ema_p.copy_(curr_p)

    return ema_multi_update


class EMAWeightAveraging(WeightAveraging):
    """Exponential Moving Average Weight Averaging using timm's ModelEmaV3.

    This callback provides advanced EMA features over standard Lightning
    EMAWeightAveraging:
    - Decay warmup: Gradually increases decay factor during early training
      for better stability
    - Step-aware decay: Supports dynamic decay scheduling based on training
      steps

    The decay warmup feature is particularly useful for models trained for many steps.
    With inv_gamma=1 and power=3/4, the decay factor reaches:
    - 0.999 at ~10K steps
    - 0.9999 at ~215.4k steps
    Args:
        device: Device to store the EMA model on. If None, uses the same device as the
            training model. Use "cpu" to save GPU memory.
        use_buffers: If True, also averages model buffers (e.g., BatchNorm statistics).
            Set to False if you plan to update batch norm statistics separately.
        decay: Base exponential decay rate. Higher values give more weight to past
            parameters. Typical values: 0.999-0.9999.
        min_decay: Minimum decay value during warmup. Usually 0.0.
        use_warmup: Enable decay warmup. The decay factor gradually increases from
            min_decay to decay over time, improving training stability.
        warmup_gamma: Warmup gamma parameter (inv_gamma in literature). Controls
            warmup speed. Default 1.0.
        warmup_power: Warmup power parameter. Controls warmup curve shape.
            - 3/4: Good for medium training (100K-500K steps)
        update_every_n_steps: Update the EMA model every N optimizer steps. Default 1.
        update_starting_at_step: Start updates after this step index (0-based).
            If None, starts immediately.
        update_starting_at_epoch: Start updates after this epoch index (0-based).
            If None, epoch-based control is disabled.

    Note:
        Like WeightAveraging, this callback doesn't support sharded models and may
    """

    def __init__(
        self,
        device: torch.device | str | int | None = None,
        use_buffers: bool = True,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        use_warmup: bool = True,
        warmup_gamma: float = 1.0,
        warmup_power: float = 3 / 4,
        update_every_n_steps: int = 1,
        update_starting_at_step: int | None = None,
        update_starting_at_epoch: int | None = None,
    ) -> None:
        # Initialize parent without avg_fn since we're using ModelEmaV3
        # Note: We can't pass use_buffers to parent since ModelEmaV3
        # handles it differently
        super().__init__(
            device=device,
            use_buffers=use_buffers,
            multi_avg_fn=get_ema_multi_avg_fn(
                decay, use_warmup, min_decay, warmup_gamma, warmup_power
            ),
        )
        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step
        self.update_starting_at_epoch = update_starting_at_epoch

    def should_update(
        self, step_idx: int | None = None, epoch_idx: int | None = None
    ) -> bool:
        """Decide when to update the model weights.

        Args:
            step_idx: The current step index.
            epoch_idx: The current epoch index.
        Returns:
            bool: True if the model weights should be updated, False otherwise.

        """
        if step_idx is not None:
            # Check step-based conditions only if we have a valid step_idx
            meets_step_requirement = (
                self.update_starting_at_step is None
                or step_idx >= self.update_starting_at_step
            )
            meets_step_frequency = (
                self.update_every_n_steps > 0
                and step_idx % self.update_every_n_steps == 0
            )
            if meets_step_requirement and meets_step_frequency:
                return True

        if epoch_idx is not None:
            # Check epoch-based condition only if we specify one
            meets_epoch_requirement = (
                self.update_starting_at_epoch is not None
                and epoch_idx >= self.update_starting_at_epoch
            )
            if meets_epoch_requirement:
                return True

        return False
