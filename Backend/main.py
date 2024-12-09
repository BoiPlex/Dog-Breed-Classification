import os
import matplotlib.pyplot as plt



def train_and_evaluate(optimizer_name, X_train, y_train, X_test, y_test, encoder, num_classes, output_dir, class_names):
    """Train and evaluate a model with a specific optimizer."""
    print(f"\nTraining model with {optimizer_name.upper()} optimizer...")

    # Recreate the model
    model = create_model(num_classes)

    # Train the model
    try:
        history = train_model(model, X_train, y_train, X_test, y_test, optimizer_name)
    except Exception as e:
        print(f"Error during training with {optimizer_name.upper()} optimizer: {e}")
        return

    # Evaluate the model
    print(f"\nEvaluating model trained with {optimizer_name.upper()} optimizer")
    try:
        evaluate_model(model, X_test, y_test, encoder, output_dir, optimizer_name, class_names)
    except Exception as e:
        print(f"Error during evaluation with {optimizer_name.upper()} optimizer: {e}")

    # Save the trained model
    model_path = os.path.join(output_dir, f"model_{optimizer_name}.keras")
    try:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model for {optimizer_name.upper()} optimizer: {e}")

    # Plot training metrics and learning rate curve
    plot_training_metrics(history, optimizer_name, output_dir)
    plot_learning_rate(history, optimizer_name, output_dir)

    return history
def plot_training_metrics(history, optimizer_name, output_dir):
    """Plot and save training metrics."""
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
    mat_file_path = "/content/drive/MyDrive/data/lists/train_list.mat"
    print(f"Loading data from {mat_file_path}...")
    X_train, X_test, y_train, y_test, encoder, num_classes, class_names = load_data_from_mat(mat_file_path)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Processed class names: {class_names}")

    output_dir = "/content/drive/MyDrive/output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory created at: {output_dir}")

    optimizers = ['adam', 'adamw', 'sgd']
    histories = []

    for optimizer_name in optimizers:
        history =  train_and_evaluate(optimizer_name, X_train, y_train, X_test, y_test, encoder, num_classes, output_dir, class_names)
        if history:
            histories.append((history, optimizer_name))

    compare_convergence(histories, output_dir)

    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()

