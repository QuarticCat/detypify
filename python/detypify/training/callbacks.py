"""Self Write Training Callbacks"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, override

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.weight_averaging import WeightAveraging

logger = logging.getLogger(__name__)

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

    @override
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        import torch

        if self.logged_batches >= self.max_batches:
            return

        # Check if outputs is available (requires test_step to return pred)
        if outputs is None:
            return

        if not isinstance(outputs, torch.Tensor):
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
            from math import ceil

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
                zip(selected_images, selected_preds, true_labels, strict=True)
            ):
                ax = axes_flat[i]

                # Image is (C, H, W), usually (1, H, W) for grayscale
                # Convert to (H, W) numpy for imshow
                img_np = img.cpu().numpy()
                if img_np.shape[0] == 1:
                    img_np = img_np.squeeze(0)

                ax.imshow(img_np, cmap="gray")

                pred_name = self.classes[pred_idx] if pred_idx < len(self.classes) else str(pred_idx.item())
                true_name = self.classes[true_idx] if true_idx < len(self.classes) else str(true_idx.item())

                # Determine color: red if wrong, green if right
                is_correct = pred_idx == true_idx
                title_color = "green" if is_correct else "red"

                ax.set_title(f"Truth: {true_name}\nPrediction: {pred_name}", color=title_color)
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

            tensorboard.add_figure(tag, fig, global_step=batch_idx)
            plt.close(fig)

        self.logged_batches += 1


class LogTestConfusionCallback(Callback):
    def __init__(
        self,
        classes: list[str],
        top_k_false_predicted_labels: int = 10,
        examples_per_label: int = 4,
        max_confusion_labels: int = 40,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.top_k_false_predicted_labels = top_k_false_predicted_labels
        self.examples_per_label = examples_per_label
        self.max_confusion_labels = max_confusion_labels
        self.confusion_matrix = None
        self.false_pred_examples = {}

    @override
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        import torch

        num_classes = len(self.classes)
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.false_pred_examples: dict[int, list[tuple[Tensor, int, int]]] = {}

    @override
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        import torch

        if outputs is None or self.confusion_matrix is None:
            return

        if not isinstance(outputs, torch.Tensor):
            return

        pred_logits = outputs
        image, label = batch["image"], batch["label"]
        preds = torch.argmax(pred_logits, dim=1)

        labels_cpu = label.detach().to("cpu", dtype=torch.int64)
        preds_cpu = preds.detach().to("cpu", dtype=torch.int64)
        valid_mask = (
            (labels_cpu >= 0) & (labels_cpu < len(self.classes)) & (preds_cpu >= 0) & (preds_cpu < len(self.classes))
        )
        if not valid_mask.any():
            return

        labels_cpu = labels_cpu[valid_mask]
        preds_cpu = preds_cpu[valid_mask]
        num_classes = len(self.classes)
        flat_indices = labels_cpu * num_classes + preds_cpu
        batch_confusion = torch.bincount(flat_indices, minlength=num_classes * num_classes)
        self.confusion_matrix += batch_confusion.reshape(num_classes, num_classes)

        wrong_mask = preds != label
        if not wrong_mask.any():
            return

        wrong_images = (image[wrong_mask].detach().to("cpu") * 255).to(dtype=torch.uint8)
        wrong_preds = preds[wrong_mask].detach().to("cpu", dtype=torch.int64)
        wrong_labels = label[wrong_mask].detach().to("cpu", dtype=torch.int64)

        for img, pred_idx, true_idx in zip(wrong_images, wrong_preds, wrong_labels, strict=True):
            pred_label_idx = int(pred_idx.item())
            if pred_label_idx < 0 or pred_label_idx >= len(self.classes):
                continue

            examples = self.false_pred_examples.setdefault(pred_label_idx, [])
            if len(examples) < self.examples_per_label:
                examples.append((img, int(true_idx.item()), pred_label_idx))

    @override
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.confusion_matrix is None:
            return

        from lightning.pytorch.loggers import TensorBoardLogger

        if not isinstance(trainer.logger, TensorBoardLogger):
            return

        import matplotlib as mpl
        import torch

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        tensorboard = trainer.logger.experiment
        confusion = self.confusion_matrix

        false_by_pred = confusion.sum(dim=0) - confusion.diag()
        top_pred_count = min(self.top_k_false_predicted_labels, int((false_by_pred > 0).sum().item()))
        if top_pred_count > 0:
            top_false_counts, top_false_pred_indices = torch.topk(false_by_pred, k=top_pred_count)
            self._log_top_false_predicted_text(tensorboard, top_false_pred_indices, top_false_counts)
            self._log_top_false_predicted_examples(tensorboard, top_false_pred_indices)

        self._log_confusion_matrix(tensorboard, confusion, false_by_pred)

        plt.close("all")

    def _log_top_false_predicted_text(self, tensorboard, top_indices, top_counts) -> None:
        lines = ["| Predicted label | False predictions |", "| --- | ---: |"]
        for pred_idx, count in zip(top_indices.tolist(), top_counts.tolist(), strict=True):
            lines.append(f"| {self.classes[pred_idx]} | {count} |")

        tensorboard.add_text("test/top_false_predicted_labels", "\n".join(lines), global_step=0)

    def _log_top_false_predicted_examples(self, tensorboard, top_indices) -> None:
        from math import ceil

        import matplotlib.pyplot as plt

        examples = []
        for pred_idx in top_indices.tolist():
            examples.extend(self.false_pred_examples.get(pred_idx, []))

        if not examples:
            return

        num_images = len(examples)
        cols = min(self.examples_per_label, num_images)
        rows = ceil(num_images / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
        axes_flat = [axes] if num_images == 1 else axes.flatten()

        for ax, (img, true_idx, pred_idx) in zip(axes_flat, examples, strict=False):
            img_np = img.numpy()
            if img_np.shape[0] == 1:
                img_np = img_np.squeeze(0)

            true_name = self.classes[true_idx] if 0 <= true_idx < len(self.classes) else str(true_idx)
            pred_name = self.classes[pred_idx] if 0 <= pred_idx < len(self.classes) else str(pred_idx)

            ax.imshow(img_np, cmap="gray")
            ax.set_title(f"Truth: {true_name}\nPrediction: {pred_name}", color="red")
            ax.axis("off")

        for ax in axes_flat[num_images:]:
            ax.axis("off")

        plt.tight_layout()
        tensorboard.add_figure("test/top_false_predicted_label_examples", fig, global_step=0)
        plt.close(fig)

    def _log_confusion_matrix(self, tensorboard, confusion, false_by_pred) -> None:
        import matplotlib.pyplot as plt
        import torch

        total_by_label = confusion.sum(dim=1) + confusion.sum(dim=0)
        error_involvement = total_by_label - (2 * confusion.diag())
        candidate_scores = torch.maximum(error_involvement, false_by_pred)
        num_labels = min(self.max_confusion_labels, len(self.classes))

        if len(self.classes) > num_labels:
            _, indices = torch.topk(candidate_scores, k=num_labels)
            indices = indices.sort().values
            matrix = confusion[indices][:, indices]
            class_names = [self.classes[i] for i in indices.tolist()]
            title = f"Test confusion matrix - top {num_labels} error-involved labels"
        else:
            matrix = confusion
            class_names = self.classes
            title = "Test confusion matrix"

        matrix_float = matrix.to(dtype=torch.float32)
        row_totals = matrix_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        normalized = matrix_float / row_totals

        fig_size = max(8, min(20, len(class_names) * 0.45))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(normalized.numpy(), interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        tensorboard.add_figure("test/confusion_matrix", fig, global_step=0)
        plt.close(fig)


def get_ema_multi_avg_fn(decay: float, min_decay: float, warmup_gamma: float, warmup_power: float, *, use_warmup: bool):
    """
    Get a multi_avg_fn applying EMA with Inverse Gamma warmup schedule,
    adapted from torch lightning's and timm's implementation

    Unlike the standard get_ema_avg_fn which uses a fixed decay, this version
    calculates the decay dynamically based on the step count (num_averaged).
    This allows you to start EMA immediately (step 0) without initialization bias
    """

    import torch

    @torch.no_grad()
    def ema_multi_update(averaged_param_list: list[Tensor], current_param_list: list[Tensor], num_averaged: Tensor):
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

        for ema_p, curr_p in zip(averaged_param_list, current_param_list, strict=True):
            if ema_p.is_floating_point() or ema_p.is_complex():
                lerp_ema_params.append(ema_p)
                lerp_curr_params.append(curr_p)
            else:
                copy_ema_params.append(ema_p)
                copy_curr_params.append(curr_p)

        # Apply Fused Update (Horizontal Fusion)
        if lerp_ema_params:
            torch._foreach_lerp_(lerp_ema_params, lerp_curr_params, weight=1.0 - cur_decay)

        # Apply Standard Copy for integers
        for ema_p, curr_p in zip(copy_ema_params, copy_curr_params, strict=True):
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

    Note:
        Like WeightAveraging, this callback doesn't support sharded models and may
        experience memory increases due to storing averaged parameters.
    """

    def __init__(
        self,
        device: device | str | int | None = None,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        warmup_gamma: float = 25.0,
        warmup_power: float = 3 / 4,
        update_every_n_steps: int = 1,
        update_starting_at_step: int | None = None,
        *,
        use_buffers: bool = True,
        use_warmup: bool = True,
    ) -> None:
        # Initialize parent without avg_fn since we're using ModelEmaV3
        # Note: We can't pass use_buffers to parent since ModelEmaV3
        # handles it differently
        super().__init__(
            device=device,
            use_buffers=use_buffers,
            multi_avg_fn=get_ema_multi_avg_fn(decay, min_decay, warmup_gamma, warmup_power, use_warmup=use_warmup),
        )
        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step

    @override
    def should_update(self, step_idx: int | None = None, epoch_idx: int | None = None) -> bool:
        """Decide when to update the model weights.

        Args:
            step_idx: The current step index.
            epoch_idx: The current epoch index.
        Returns:
            bool: True if the model weights should be updated, False otherwise.

        """
        return (
            step_idx is not None
            and epoch_idx is None
            and (self.update_starting_at_step is None or step_idx >= self.update_starting_at_step)
            and self.update_every_n_steps > 0
            and step_idx % self.update_every_n_steps == 0
        )


class ExportBestModelToONNX(Callback):
    """Export the best model checkpoint to ONNX format after training completes.

    This callback finds the best checkpoint saved during training and exports it
    to ONNX format, making it ready for deployment.

    Args:
        onnx_dir: Directory where ONNX file will be saved
        model_name: Name to use for the ONNX file (without extension)
        checkpoint_callback: The ModelCheckpoint callback used during training.
        use_compile: Whether training used torch.compile. Also controls ONNX export optimization.
        dynamo: Whether to use torch.dynamo for ONNX export (default: True)
        external_data: Whether to save weights as external data (default: False)
    """

    def __init__(
        self,
        save_dir: Path,
        model_name: str,
        checkpoint_callback: ModelCheckpoint,
        *,
        use_compile: bool,
        dynamo: bool = True,
        external_data: bool = False,
    ) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.checkpoint_callback = checkpoint_callback
        self.use_compile = use_compile
        self.dynamo = dynamo
        self.external_data = external_data

    @override
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Export the best model to ONNX when training finishes."""
        checkpoint_callback = self.checkpoint_callback
        # Get the best model path
        best_model_path = Path(checkpoint_callback.best_model_path)
        if not best_model_path.exists():
            logger.warning("No best model checkpoint available. Skipping ONNX export.")
            return

        logger.info("Loading best checkpoint from: %s", best_model_path)

        # Load the best checkpoint
        from detypify.training.model import MobileNetModel

        best_model = MobileNetModel.load_from_checkpoint(best_model_path, model_name=self.model_name)
        # Freeze and prepare model for export
        best_model.freeze()
        if hasattr(best_model, "use_compile"):
            best_model.use_compile = False  # type: ignore

        # Create ONNX directory
        save_path = self.save_dir / f"{best_model_path.stem}.onnx"

        logger.info("Exporting best model to ONNX: %s", save_path)

        # Export to ONNX. The dynamo exporter can fail on some PyTorch/timm
        # operator combinations; keep the training run successful by falling
        # back to the legacy exporter.
        try:
            best_model.to_onnx(
                save_path,
                best_model.example_input_array,
                dynamo=self.dynamo,
                external_data=self.external_data,
                optimize=self.use_compile,
            )
        except Exception:
            if not self.dynamo:
                raise
            logger.exception("Dynamo ONNX export failed. Retrying with dynamo=False.")
            best_model.to_onnx(
                save_path,
                best_model.example_input_array,
                dynamo=False,
                external_data=self.external_data,
                optimize=False,
            )

        logger.info("Successfully exported best model to: %s", save_path)
