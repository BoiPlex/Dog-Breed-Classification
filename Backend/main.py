from data_loader import load_data_with_split, load_data_from_mat
from model import create_model
from train import train_model
from evaluate import evaluate_model
import matplotlib.pyplot as plt

# 1. load data : `load_data_with_split` or `load_data_from_mat`
use_mat_data = True  # if True use `load_data_from_mat`,else use `load_data_with_split`

if use_mat_data:
    X_train, X_test, y_train, y_test, encoder, num_classes = load_data_from_mat()
else:
    X_train, X_test, y_train, y_test, encoder, num_classes = load_data_with_split()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("Number of classes:", num_classes)

# 2. creat model
model = create_model(num_classes)

# 3. train model for three optimizer
histories = train_model(model, X_train, y_train, X_test, y_test)

# 4. eval and visualization
for name, history in histories.items():
    print(f"\nEvaluating model trained with {name} optimizer")
    evaluate_model(model, X_test, y_test, encoder)
    
    
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
    plt.show()
