"""
Experiment: Mask Quality vs Model Performance

This script trains multiple models while degrading the training masks by
randomly removing connected components (particles) and optionally adding
pixel-level noise. It helps answer how robust the model is to incomplete
or noisy annotations.
"""

import os
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from src.model.SegmentationDataset import SegmentationDataset
from src.model.UNet import UNet
from src.model.ModelEvaluator import ModelEvaluator
from src.model.DataTools import slice_dataset_in_four, process_and_slice, get_normalizer
from src.model.DataAugmenter import DataAugmenter


class MaskQualityExperiment:
    def __init__(self, images_path, masks_path, output_dir="data/experiments/mask_quality"):
        self.images_path = images_path
        self.masks_path = masks_path
        self.output_dir = output_dir
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        os.makedirs(output_dir, exist_ok=True)

    def _clone_subset(self, dataset, subset_indices):
        """Create an independent SegmentationDataset copy for the given indices."""
        images, masks, filenames = [], [], []
        for idx in subset_indices:
            img, mask = dataset[idx]
            images.append(img.clone())
            masks.append(mask.clone())
            if hasattr(dataset, "image_filenames"):
                filenames.append(dataset.image_filenames[idx])
            else:
                filenames.append(f"sample_{idx}")
        return SegmentationDataset.from_image_set(images, masks, filenames)

    def prepare_data_splits(self, train_split=0.6, val_split=0.2, test_split=0.2, random_seed=42, input_size=(256, 256)):
        """
        Prepare fixed train/val/test splits (on sliced patches) used across all corruption levels.
        """
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        dataset = SegmentationDataset(self.images_path, self.masks_path)
        dataset_sliced = slice_dataset_in_four(dataset, input_size)
        total_size = len(dataset_sliced)
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        test_size = int(total_size * test_split)
        val_size = int(total_size * val_split)

        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size + val_size]
        train_indices = indices[test_size + val_size:]

        self.dataset_sliced = dataset_sliced
        self.train_indices = train_indices.tolist()
        self.val_indices = val_indices.tolist()
        self.test_indices = test_indices.tolist()

        print(f"Sliced patches: {total_size}")
        print(f"Train patches: {len(self.train_indices)}, Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")

    def create_dataloaders(self, component_drop_fraction, min_components_to_keep, noise_fraction, noise_white_probability, input_size=(256, 256), with_augmentation=True, batch_size=8, seed=None):
        """
        Build dataloaders for a specific corruption configuration.
        """
        # Clone train subset to avoid cross-run contamination
        train_dataset = self._clone_subset(self.dataset_sliced, self.train_indices)

        # Permanently corrupt the training masks for this run
        DataAugmenter.corrupt_masks(
            train_dataset,
            component_drop_fraction=component_drop_fraction,
            min_components_to_keep=min_components_to_keep,
            noise_fraction=noise_fraction,
            noise_white_probability=noise_white_probability,
            seed=seed,
        )

        # Wrap in Subset to align with augmenter expectations
        train_subset = Subset(train_dataset, list(range(len(train_dataset))))

        augmenter = DataAugmenter()
        if with_augmentation:
            train_data = augmenter.augment_dataset(train_subset)
        else:
            train_data = augmenter.augment_dataset(train_subset, [False, False, False, False, False, False, False])

        # Validation and test use the original (uncorrupted) data
        val_subset = Subset(self.dataset_sliced, self.val_indices)
        test_subset = Subset(self.dataset_sliced, self.test_indices)
        val_data = process_and_slice(val_subset, input_size)
        test_data = process_and_slice(test_subset, input_size)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        return train_loader, val_loader, test_loader

    def train_single_model(self, drop_fraction, min_keep, noise_fraction, noise_white_probability, epochs=100, learning_rate=0.0001, input_size=(256, 256), with_augmentation=True, seed=None):
        """
        Train a single model for the given corruption settings and return metrics.
        """
        print(f"\n{'=' * 80}")
        print(f"Mask corruption: drop {int(drop_fraction * 100)}% components, min keep {min_keep}, noise {noise_fraction:.3f} (p_white={noise_white_probability})")
        print(f"{'=' * 80}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        train_loader, val_loader, test_loader = self.create_dataloaders(
            component_drop_fraction=drop_fraction,
            min_components_to_keep=min_keep,
            noise_fraction=noise_fraction,
            noise_white_probability=noise_white_probability,
            input_size=input_size,
            with_augmentation=with_augmentation,
            seed=seed,
        )

        unet = UNet().to(device)
        unet.preferred_input_size = input_size
        unet.normalizer = get_normalizer(train_loader.dataset.dataset)

        model_name = f"UNet_maskquality_drop{int(drop_fraction*100)}_noise{int(noise_fraction*1000)}_{self.timestamp}.pt"

        start_time = datetime.datetime.now()
        unet.train_model(
            training_dataloader=train_loader,
            validation_dataloader=val_loader,
            epochs=epochs,
            learningRate=learning_rate,
            model_name=model_name,
            cross_validation="holdout",
            with_early_stopping=True,
            loss_function="combined",
            scheduler_type="none",
        )
        training_time = (datetime.datetime.now() - start_time).total_seconds()

        evaluation_result = ModelEvaluator.evaluate_model(unet, test_loader)

        result = {
            "drop_fraction": drop_fraction,
            "noise_fraction": noise_fraction,
            "noise_white_probability": noise_white_probability,
            "mean_iou": evaluation_result.mean_iou,
            "mean_dice": evaluation_result.mean_dice,
            "std_iou": float(np.std(evaluation_result.iou_scores)),
            "std_dice": float(np.std(evaluation_result.dice_scores)),
            "min_iou": evaluation_result.min_iou,
            "max_iou": evaluation_result.max_iou,
            "min_dice": evaluation_result.min_dice,
            "max_dice": evaluation_result.max_dice,
            "training_time": training_time,
            "model_name": model_name,
        }

        print(f"\nResults: IoU {result['mean_iou']:.4f} +/- {result['std_iou']:.4f}, Dice {result['mean_dice']:.4f} +/- {result['std_dice']:.4f}")
        return result

    def run_experiment(self, drop_fractions, min_components_to_keep=0, noise_fraction=0.0, noise_white_probability=0.5, epochs=100, learning_rate=0.0001, input_size=(256, 256), with_augmentation=True, random_seed=42):
        """
        Run the mask quality experiment over a list of component drop fractions.
        """
        print(f"\n{'=' * 80}")
        print("MASK QUALITY EXPERIMENT")
        print(f"{'=' * 80}")
        print(f"Drop fractions: {[int(df*100) for df in drop_fractions]}%")
        print(f"Noise fraction: {noise_fraction} (p_white={noise_white_probability})")
        print(f"Epochs: {epochs}, LR: {learning_rate}, Augmentation: {with_augmentation}")

        self.prepare_data_splits(random_seed=random_seed, input_size=input_size)

        all_results = []
        for drop_fraction in drop_fractions:
            result = self.train_single_model(
                drop_fraction=drop_fraction,
                min_keep=min_components_to_keep,
                noise_fraction=noise_fraction,
                noise_white_probability=noise_white_probability,
                epochs=epochs,
                learning_rate=learning_rate,
                input_size=input_size,
                with_augmentation=with_augmentation,
                seed=random_seed,
            )
            all_results.append(result)

        self.save_results(all_results)
        self.plot_results(all_results)
        return all_results

    def save_results(self, results):
        """Persist results to a text file."""
        results_file = os.path.join(self.output_dir, f"results_{self.timestamp}.txt")
        with open(results_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MASK QUALITY EXPERIMENT RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Images path: {self.images_path}\n")
            f.write(f"Masks path: {self.masks_path}\n\n")

            f.write("RESULTS SUMMARY:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Drop %':<10} {'Mean IoU':<18} {'Mean Dice':<18} {'Time (s)':<12}\n")
            f.write("-" * 80 + "\n")
            for r in results:
                f.write(f"{int(r['drop_fraction']*100):<10}"
                        f"{r['mean_iou']:.4f}±{r['std_iou']:.4f}  "
                        f"{r['mean_dice']:.4f}±{r['std_dice']:.4f}  "
                        f"{r['training_time']:.2f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED RESULTS:\n")
            f.write("=" * 80 + "\n\n")
            for r in results:
                f.write(f"Drop fraction: {r['drop_fraction']:.2f}, noise {r['noise_fraction']:.3f} (p_white={r['noise_white_probability']})\n")
                f.write(f"Model: {r['model_name']}\n")
                f.write(f"  IoU:  {r['mean_iou']:.4f} ± {r['std_iou']:.4f} [{r['min_iou']:.4f}, {r['max_iou']:.4f}]\n")
                f.write(f"  Dice: {r['mean_dice']:.4f} ± {r['std_dice']:.4f} [{r['min_dice']:.4f}, {r['max_dice']:.4f}]\n")
                f.write(f"  Training Time: {r['training_time']:.2f} seconds\n")
                f.write("-" * 80 + "\n")

        print(f"\nResults saved to: {results_file}")

    def plot_results(self, results):
        """Plot performance vs. corruption level."""
        drop_pct = [r["drop_fraction"] * 100 for r in results]
        mean_iou = [r["mean_iou"] for r in results]
        std_iou = [r["std_iou"] for r in results]
        mean_dice = [r["mean_dice"] for r in results]
        std_dice = [r["std_dice"] for r in results]
        times = [r["training_time"] for r in results]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Mask Corruption vs Model Performance", fontsize=16, fontweight="bold")

        ax1 = axes[0, 0]
        ax1.errorbar(drop_pct, mean_iou, yerr=std_iou, marker="o", linewidth=2, capsize=4)
        ax1.set_xlabel("Component drop (%)")
        ax1.set_ylabel("Mean IoU")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("IoU vs Mask Corruption")

        ax2 = axes[0, 1]
        ax2.errorbar(drop_pct, mean_dice, yerr=std_dice, marker="s", color="orange", linewidth=2, capsize=4)
        ax2.set_xlabel("Component drop (%)")
        ax2.set_ylabel("Mean Dice")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Dice vs Mask Corruption")

        ax3 = axes[1, 0]
        ax3.plot(drop_pct, times, marker="D", color="green", linewidth=2)
        ax3.set_xlabel("Component drop (%)")
        ax3.set_ylabel("Training Time (s)")
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Training Time vs Mask Corruption")

        ax4 = axes[1, 1]
        ax4.plot(drop_pct, mean_iou, marker="o", linewidth=2, label="IoU")
        ax4.plot(drop_pct, mean_dice, marker="s", linewidth=2, label="Dice")
        ax4.set_xlabel("Component drop (%)")
        ax4.set_ylabel("Score")
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_title("Scores vs Mask Corruption")

        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, f"plots_{self.timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Plots saved to: {plot_file}")
        plt.show()


def main():
    images_path = "data/medres_images"
    masks_path = "data/medres_masks"

    drop_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    noise_fraction = 0.0  # set >0.0 to add random pixel noise
    noise_white_probability = 0.5

    epochs = 50
    learning_rate = 0.0001
    input_size = (256, 256)
    with_augmentation = True
    random_seed = 42

    experiment = MaskQualityExperiment(images_path, masks_path)
    experiment.run_experiment(
        drop_fractions=drop_fractions,
        min_components_to_keep=1,
        noise_fraction=noise_fraction,
        noise_white_probability=noise_white_probability,
        epochs=epochs,
        learning_rate=learning_rate,
        input_size=input_size,
        with_augmentation=with_augmentation,
        random_seed=random_seed,
    )


if __name__ == "__main__":
    main()
