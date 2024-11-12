from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.experimental import AdamW

def train_model(model, X_train, y_train, X_test, y_test):
    optimizers = {
        'adam': Adam(learning_rate=0.001),
        'adamw': AdamW(learning_rate=0.001),
        'sgd': SGD(learning_rate=0.001, momentum=0.9)
    }
    
    histories = {}  #  store the history for every optimizer 

    for name, optimizer in optimizers.items():
        print(f"Training with {name} optimizer")
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        early_stopping = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3, verbose=1)

        history = model.fit(
            X_train, y_train, 
            batch_size=64, 
            epochs=50, 
            validation_data=(X_test, y_test), 
            callbacks=[early_stopping, reduce_lr]
        )

        histories[name] = history  # store the current optimizer trained result
    return histories
