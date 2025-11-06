
def plot_loss(training_loss_values: list[float], validation_loss_values: list[float]) -> None:
    import numpy as np

    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    try:
        plt.clf()
        plt.yscale('log')
        plt.plot(training_loss_values, label='Training Loss', color='blue', linestyle='-')
        plt.plot(validation_loss_values, label='Validation Loss', color='red', linestyle='--')
        plt.xticks(np.arange(len(training_loss_values)), np.arange(1, len(training_loss_values)+1))
        plt.legend(loc="upper right")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.pause(0.1)  # pause to update the plot
    except Exception as e:
        print("Error when plotting!")
        print(e.with_traceback)

def plot_difference(input, prediction, label, iou, dice_score):
        import matplotlib
        import numpy as np

        matplotlib.use('QtAgg')
        import matplotlib.pyplot as plt

        prediction_uint8 = (np.array(prediction) * 255).astype(np.uint8).squeeze(0)
        label_uint8 = (np.array(label) * 255).astype(np.uint8).squeeze(0)
        input_uint8 = (np.array(input) * 255).astype(np.uint8).squeeze()

        false_positives = ((prediction_uint8 == 255) & (label_uint8 == 0))  # FP: Red
        false_negatives = ((prediction_uint8 == 0) & (label_uint8 == 255))  # FN: Blue
        true_positives = ((prediction_uint8 == 255) & (label_uint8 == 255)).astype(np.uint8)  # TP: White

        overlay = np.zeros((*false_positives.shape, 3), dtype=np.uint8)

        overlay[..., 0] = false_positives * 255
        overlay[..., 2] = false_negatives * 255  # Blue channel for FN
        overlay[..., 0] += true_positives * 255
        overlay[..., 1] += true_positives * 255
        overlay[..., 2] += true_positives * 255
        fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharex=True, sharey=True)

        axes[0].imshow(input_uint8, cmap='gray')
        axes[0].set_title("Input")

        axes[1].imshow(prediction_uint8, cmap='gray')
        axes[1].set_title("Prediction")

        # axes[2].imshow(label_uint8, cmap='gray')
        # axes[2].set_title("Label")

        axes[2].imshow(overlay)
        axes[2].set_title("Difference (FP: Red, FN: Blue)")

        fig.text(0.5, 0.95, f"IoU: {iou:.2f}   Dice Score: {dice_score:.2f}",
         ha='center', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()