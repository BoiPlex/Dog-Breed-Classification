from data_loader import load_data_from_mat
from model import create_model
from train import train_model
from evaluate import evaluate_model
import matplotlib.pyplot as plt
import os

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
    optimizers = ['adam', 'adamw', 'sgd']  

    for optimizer_name in optimizers:
        print(f"\nTraining model with {optimizer_name.upper()} optimizer...")

        # Recreate model for each optimizer
        model = create_model(num_classes)

        # Train model
        try:
            history = train_model(model, X_train, y_train, X_test, y_test, optimizer_name)
        except Exception as e:
            print(f"Error during training with {optimizer_name.upper()} optimizer: {e}")
            continue

        # Evaluate model
        print(f"\nEvaluating model trained with {optimizer_name.upper()} optimizer")
        try:
            evaluate_model(model, X_test, y_test, encoder)
        except Exception as e:
            print(f"Error during evaluation with {optimizer_name.upper()} optimizer: {e}")
            continue

        # Save model
        model_path = os.path.join(output_dir, f"model_{optimizer_name}.h5")
        try:
            model.save(model_path)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error saving model for {optimizer_name.upper()} optimizer: {e}")
            continue

        # Plot training metrics
        try:
            plt.figure(figsize=(12, 5))
            # Accuracy curve
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{optimizer_name.upper()} - Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            # Loss curve
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'{optimizer_name.upper()} - Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.suptitle(f"Training Metrics for {optimizer_name.upper()} Optimizer")
            plot_path = os.path.join(output_dir, f"{optimizer_name}_metrics.png")
            plt.savefig(plot_path)
            print(f"Metrics plot saved to {plot_path}")
            plt.show()
        except Exception as e:
            print(f"Error saving plot for {optimizer_name.upper()} optimizer: {e}")

    print("\nTraining and evaluation completed.")

if __name__ == "__main__":
    main()
