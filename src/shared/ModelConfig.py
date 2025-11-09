class ModelConfig():
    def __init__(self, images_path, masks_path, epochs, learning_rate, with_early_stopping, with_data_augmentation, test_images_path = None, test_masks_path = None):
        self.images_path: str = images_path
        self.masks_path: str = masks_path
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.with_early_stopping: bool = with_early_stopping
        self.with_data_augmentation: bool = with_data_augmentation
        self.test_images_path: str = test_images_path
        self.test_masks_path: str = test_masks_path
        self.train_subset_size = 0.6
        self.validation_subset_size = 0.2
