import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test, encoder):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss on the test set: {loss:.2f}")
    print(f"Accuracy on the test set: {acc:.3f}")

    predictions = model.predict(X_test)
    label_predictions = encoder.inverse_transform(predictions)

    rows, cols = 5, 3
    fig, ax = plt.subplots(rows, cols, figsize=(25, 25))
    for i in range(rows):
        for j in range(cols):
            index = np.random.randint(0, len(X_test))
            ax[i, j].imshow(X_test[index])
            ax[i, j].set_title(f'Predicted: {label_predictions[index]}\nActually: {encoder.inverse_transform(y_test)[index]}')
    plt.tight_layout()
    plt.show()
