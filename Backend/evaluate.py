import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os

def evaluate_model(model, X_test, y_test, encoder, output_dir, optimizer_name, class_names):
    """Evaluate the model and save evaluation-related visualizations."""
    try:
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Loss on the test set: {loss:.2f}")
        print(f"Accuracy on the test set: {acc:.3f}")

        predictions = model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)

        if len(class_names) != y_test.shape[1]:
            raise ValueError("The number of class names does not match the number of output classes!")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(20, 15))

        sns.heatmap(
            cm,
            annot=False,
            fmt="d",
            cmap="viridis",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True
        )

        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)

        plt.title(f"Confusion Matrix - {optimizer_name.upper()}", fontsize=16)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)

        plt.tight_layout()
        confusion_matrix_path = os.path.join(output_dir, f"confusion_matrix_{optimizer_name}.png")
        plt.savefig(confusion_matrix_path, dpi=300)
        print(f"Confusion matrix heatmap saved to {confusion_matrix_path}")

        plt.show()

    except Exception as e:
        print(f"Error during evaluation: {e}")


def plot_learning_rate(history, optimizer_name, output_dir):
    """Plot learning rate schedule."""
    lr_history = history.history.get('lr', [])
    if lr_history:
        plt.figure(figsize=(10, 5))
        plt.plot(lr_history, label='Learning Rate')
        plt.title(f'Learning Rate Schedule - {optimizer_name.upper()}')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()
        lr_path = os.path.join(output_dir, f"lr_schedule_{optimizer_name}.png")
        plt.savefig(lr_path)
        print(f"Learning rate schedule plot saved to {lr_path}")
        plt.show()


def compare_convergence(histories, output_dir):
    """Compare validation accuracy for multiple optimizers."""
    try:
        plt.figure(figsize=(12, 6))
        for history, optimizer_name in histories:
            plt.plot(history.history['val_accuracy'], label=f'{optimizer_name.upper()}')
        plt.title('Convergence Speed Comparison')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        convergence_path = os.path.join(output_dir, "convergence_comparison.png")
        plt.savefig(convergence_path)
        print(f"Convergence comparison plot saved to {convergence_path}")
        plt.show()
    except Exception as e:
        print(f"Error during convergence comparison: {e}")
