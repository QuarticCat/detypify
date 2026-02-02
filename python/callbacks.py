"""Self Write Training Callbacks"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.weight_averaging import WeightAveraging
from model import CNNModel, TimmModel

if TYPE_CHECKING:
    from typing import Literal

    from lightning import LightningModule, Trainer
    from torch import Tensor, device


class LogPredictCallback(Callback):
    def __init__(
        self,
        classes: list[str],
        max_batches: int = 16,
        log_type: Literal["wrong", "right", "both"] = "both",
    ) -> None:
        super().__init__()
        self.classes = classes
        self.max_batches = max_batches
        self.log_type = log_type
        self.logged_batches = 0

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        import torch

        if self.logged_batches >= self.max_batches:
            return

        # Check if outputs is available (requires test_step to return pred)
        if outputs is None:
            return

        pred_logits = outputs
        image, label = batch["image"], batch["label"]

        preds = torch.argmax(pred_logits, dim=1)

        # Identify guesses based on log_type
        if self.log_type == "wrong":
            mask = preds != label
        elif self.log_type == "right":
            mask = preds == label
        else:  # "both"
            # Create a mask of all True with same shape as label
            mask = torch.ones_like(label, dtype=torch.bool)

        if not mask.any():
            return

        # images transformed as float32, converting back
        selected_images: Tensor = image[mask] * 255
        selected_images = selected_images.to(dtype=torch.uint8)
        selected_preds = preds[mask]
        true_labels = label[mask]

        # Limit the number of images to log per batch (safety cap at 16)
        num_to_log = min(len(selected_images), 16)
        selected_images = selected_images[:num_to_log]
        selected_preds = selected_preds[:num_to_log]
        true_labels = true_labels[:num_to_log]

        from lightning.pytorch.loggers import TensorBoardLogger

        if isinstance(trainer.logger, TensorBoardLogger):
            import matplotlib as mpl
            import matplotlib.pyplot as plt

            mpl.use("Agg")

            tensorboard = trainer.logger.experiment

            # Create a grid of plots using matplotlib
            num_images = len(selected_images)
            cols = ceil(num_images**0.5)
            rows = ceil(num_images / cols)

            # Adjust figure size based on grid
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

            # Normalize axes to be iterable even if single plot
            axes_flat = [axes] if num_images == 1 else axes.flatten()

            for i, (img, pred_idx, true_idx) in enumerate(
                zip(selected_images, selected_preds, true_labels)
            ):
                ax = axes_flat[i]

                # Image is (C, H, W), usually (1, H, W) for grayscale
                # Convert to (H, W) numpy for imshow
                img_np = img.cpu().numpy()
                if img_np.shape[0] == 1:
                    img_np = img_np.squeeze(0)

                ax.imshow(img_np, cmap="gray")

                pred_name = (
                    self.classes[pred_idx]
                    if pred_idx < len(self.classes)
                    else str(pred_idx.item())
                )
                true_name = (
                    self.classes[true_idx]
                    if true_idx < len(self.classes)
                    else str(true_idx.item())
                )

                # Determine color: red if wrong, green if right
                is_correct = pred_idx == true_idx
                title_color = "green" if is_correct else "red"

                ax.set_title(
                    f"Truth: {true_name}\nPrediction: {pred_name}", color=title_color
                )
                ax.axis("off")

            # Hide unused subplots
            for i in range(num_images, len(axes_flat)):
                axes_flat[i].axis("off")

            plt.tight_layout()

            # Determine tag name
            if self.log_type == "wrong":
                tag = "wrong_predictions"
            elif self.log_type == "right":
                tag = "right_predictions"
            else:
                tag = "predictions"

            tensorboard.add_figure(
                tag,
                fig,
                global_step=batch_idx,
            )
            plt.close(fig)

        self.logged_batches += 1


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

    import torch

    @torch.no_grad()
    def ema_multi_update(
        averaged_param_list: list[Tensor],
        current_param_list: list[Tensor],
        num_averaged: Tensor,
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
        device: device | str | int | None = None,
        use_buffers: bool = True,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        use_warmup: bool = True,
        warmup_gamma: float = 25.0,
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


class ExportBestModelToONNX(Callback):
    """Export the best model checkpoint to ONNX format after training completes.

    This callback finds the best checkpoint saved during training and exports it
    to ONNX format, making it ready for deployment.

    Args:
        onnx_dir: Directory where ONNX file will be saved
        model_name: Name to use for the ONNX file (without extension)
        checkpoint_callback: The ModelCheckpoint callback used during training.
            If None, the callback will try to find it automatically.
        dynamo: Whether to use torch.dynamo for ONNX export (default: True)
        external_data: Whether to save weights as external data (default: False)
    """

    def __init__(
        self,
        onnx_dir: Path | str,
        model_name: str,
        checkpoint_callback: ModelCheckpoint | None = None,
        dynamo: bool = True,
        external_data: bool = False,
    ) -> None:
        super().__init__()
        self.onnx_dir = Path(onnx_dir)
        self.model_name = model_name
        self.checkpoint_callback = checkpoint_callback
        self.dynamo = dynamo
        self.external_data = external_data

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Export the best model to ONNX when training finishes."""
        # Find the checkpoint callback if not provided
        checkpoint_callback = self.checkpoint_callback
        if checkpoint_callback is None:
            for callback in trainer.callbacks:  # type: ignore
                if isinstance(callback, ModelCheckpoint):
                    checkpoint_callback = callback
                    break

        if checkpoint_callback is None:
            print("Warning: No ModelCheckpoint callback found. Skipping ONNX export.")
            return

        # Get the best model path
        best_model_path = checkpoint_callback.best_model_path
        if not best_model_path:
            print("Warning: No best model checkpoint available. Skipping ONNX export.")
            return

        print(f"Loading best checkpoint from: {best_model_path}")

        # Load the best checkpoint
        model_type = type(pl_module)
        if model_type == CNNModel:
            best_model = model_type.load_from_checkpoint(best_model_path)
        elif model_type == TimmModel:
            best_model = model_type.load_from_checkpoint(
                best_model_path, model_name=self.model_name
            )

        # Freeze and prepare model for export
        best_model.freeze()
        if hasattr(best_model, "use_compile"):
            best_model.use_compile = False  # type: ignore

        # Create ONNX directory
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.onnx_dir / f"{self.model_name}.onnx"

        print(f"Exporting best model to ONNX: {save_path}")

        # Export to ONNX
        best_model.to_onnx(
            save_path,
            best_model.example_input_array,
            dynamo=self.dynamo,
            external_data=self.external_data,
        )

        print(f"Successfully exported best model to: {save_path}")
