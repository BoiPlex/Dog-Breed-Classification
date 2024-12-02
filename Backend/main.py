from data_loader import load_data_from_mat
from model import create_model
from train import train_model
from evaluate import evaluate_model, plot_learning_rate
import matplotlib.pyplot as plt
import os


def train_and_evaluate(optimizer_name, X_train, y_train, X_test, y_test, encoder, num_classes, output_dir):
    print(f"\nTraining model with {optimizer_name.upper()} optimizer...")

    model = create_model(num_classes)

    try:
        history = train_model(model, X_train, y_train, X_test, y_test, optimizer_name)
    except Exception as e:
        print(f"Error during training with {optimizer_name.upper()} optimizer: {e}")
        return history  # Return history even on error for comparison

    print(f"\nEvaluating model trained with {optimizer_name.upper()} optimizer")
    try:
        evaluate_model(model, X_test, y_test, encoder, output_dir, optimizer_name)
    except Exception as e:
        print(f"Error during evaluation with {optimizer_name.upper()} optimizer: {e}")

    model_path = os.path.join(output_dir, f"model_{optimizer_name}.h5")
    try:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model for {optimizer_name.upper()} optimizer: {e}")

    plot_training_metrics(history, optimizer_name, output_dir)
    plot_learning_rate(history, optimizer_name, output_dir)

    return history

def plot_training_metrics(history, optimizer_name, output_dir):
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy', linestyle='--')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='-')
        plt.title(f'{optimizer_name.upper()} - Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss', linestyle='--')
        plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='-')
        plt.title(f'{optimizer_name.upper()} - Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.suptitle(f"Training Metrics for {optimizer_name.upper()} Optimizer")
        metrics_path = os.path.join(output_dir, f"{optimizer_name}_metrics.png")
        plt.tight_layout()
        plt.savefig(metrics_path)
        print(f"Metrics plot saved to {metrics_path}")
        plt.show()
    except Exception as e:
        print(f"Error saving plot for {optimizer_name.upper()} optimizer: {e}")



def main():
    # 1. Load data
    mat_file_path = "/Users/chinghaochang/project-170/data/lists/train_list.mat"
    print(f"Loading data from {mat_file_path}...")
    X_train, X_test, y_train, y_test, encoder, num_classes = load_data_from_mat(mat_file_path)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Number of classes: {num_classes}")

    # 2. Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory created at: {output_dir}")

    # 3. Train and evaluate models with multiple optimizers
    optimizers = ['adam', 'adamw', 'sgd']  # Define optimizer names
    histories = []  # Store histories for convergence comparison

    for optimizer_name in optimizers:
        history = train_and_evaluate(optimizer_name, X_train, y_train, X_test, y_test, encoder, num_classes, output_dir)
        if history:
            histories.append((history, optimizer_name))

    # Compare convergence speed
    compare_convergence(histories, output_dir)

    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()
