from data_loader import load_data_with_split, load_data_from_mat
from model import create_model
from train import train_model
from evaluate import evaluate_model
import matplotlib.pyplot as plt
import os

# 1. load data: Choose between `load_data_with_split` or `load_data_from_mat`
use_mat_data = True  # if True, use `load_data_from_mat`; else use `load_data_with_split`

if use_mat_data:
    X_train, X_test, y_train, y_test, encoder, num_classes = load_data_from_mat()
else:
    X_train, X_test, y_train, y_test, encoder, num_classes = load_data_with_split()

# print shapes for debugging
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("Number of classes:", num_classes)

# 2. Create model
model = create_model(num_classes)

# 3. Train model using three optimizers
print("\nStarting training with multiple optimizers...")
histories = train_model(model, X_train, y_train, X_test, y_test)

# Create output directory for saving results
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"\nOutput directory created at: {output_dir}")

# 4. results
for name, history in histories.items():
    print(f"\nEvaluating model trained with {name} optimizer")
    evaluate_model(model, X_test, y_test, encoder)
    
    # save model
    model_path = os.path.join(output_dir, f"model_{name}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # plot training metrics
    plt.figure(figsize=(12, 5))
    
    # accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # loss curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.suptitle(f"Training and Validation Metrics using {name.upper()} Optimizer")
    plot_path = os.path.join(output_dir, f"{name}_metrics.png")
    plt.savefig(plot_path)
    print(f"Metrics plot saved to {plot_path}")
    plt.show()

print("\nTraining and evaluation completed.")
