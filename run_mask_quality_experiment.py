"""
Quick start runner for the mask quality experiment.
Adjust parameters below and run this file to launch the experiment.
"""

from mask_quality_experiment import MaskQualityExperiment

# Dataset paths
IMAGES_PATH = "data/medres_images"
MASKS_PATH = "data/medres_masks"
OUTPUT_DIR = "data/experiments/mask_quality"

# Corruption settings
DROP_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # percentage of components to drop
MIN_COMPONENTS_TO_KEEP = 1                       # avoid empty masks
NOISE_FRACTION = 0.0                             # fraction of pixels to randomize
NOISE_WHITE_PROBABILITY = 0.5                    # probability randomized pixels become foreground

# Training hyperparameters
EPOCHS = 150
LEARNING_RATE = 0.0001
INPUT_SIZE = (256, 256)
WITH_AUGMENTATION = True
RANDOM_SEED = 42


if __name__ == "__main__":
    print("=" * 80)
    print("STARTING MASK QUALITY EXPERIMENT")
    print("=" * 80)
    print(f"Drop fractions: {[int(df*100) for df in DROP_FRACTIONS]}%")
    print(f"Noise fraction: {NOISE_FRACTION} (p_white={NOISE_WHITE_PROBABILITY})")
    print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Augmentation: {WITH_AUGMENTATION}")
    print("=" * 80)

    experiment = MaskQualityExperiment(
        images_path=IMAGES_PATH,
        masks_path=MASKS_PATH,
        output_dir=OUTPUT_DIR,
    )

    experiment.run_experiment(
        drop_fractions=DROP_FRACTIONS,
        min_components_to_keep=MIN_COMPONENTS_TO_KEEP,
        noise_fraction=NOISE_FRACTION,
        noise_white_probability=NOISE_WHITE_PROBABILITY,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        input_size=INPUT_SIZE,
        with_augmentation=WITH_AUGMENTATION,
        random_seed=RANDOM_SEED,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Check results in: {OUTPUT_DIR}")
