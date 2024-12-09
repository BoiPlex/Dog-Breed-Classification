from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        logs = logs or {}
        logs['learning_rate'] = lr
        print(f"Learning rate at end of epoch {epoch + 1}: {lr}")

def train_model(model, X_train, y_train, X_test, y_test, optimizer_name, batch_size=128, epochs=500):

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

    optimizers = {
        'adam': Adam(learning_rate=0.0001),
        'adamw': AdamW(learning_rate=0.0001),
        'sgd': SGD(learning_rate=0.001, momentum=0.9)
    }

    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    optimizer = optimizers[optimizer_name]
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"\nTraining with {optimizer_name.upper()} optimizer...")

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr, LearningRateLogger()]
    )

    return history
