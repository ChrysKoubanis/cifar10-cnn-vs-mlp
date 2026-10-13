import time
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


def MLP_model():
    """
    Preprocesses data for 1D input and trains the Multi-Layer Perceptron (MLP) baseline model.
    """
    # Preprocess and flatten image arrays (32x32x3 = 3072 features)
    x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

    y_train_flat = y_train.ravel()
    y_test_flat = y_test.ravel()

    print("\n==========================================")
    print("Building and Training the MLP Model...")
    print("==========================================")
    
    model_mlp = keras.Sequential([
        keras.layers.Input(shape=(3072,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model_mlp.summary()

    model_mlp.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    NUM_EPOCHS = 10
    mlp_start_time = time.time()
    
    history_mlp = model_mlp.fit(
        x_train_flat, 
        y_train_flat, 
        epochs=NUM_EPOCHS, 
        validation_data=(x_test_flat, y_test_flat)
    )
    
    mlp_time_taken = time.time() - mlp_start_time
    
    # Evaluate MLP metrics
    final_mlp_train_acc = history_mlp.history['accuracy'][-1]
    final_mlp_val_acc = history_mlp.history['val_accuracy'][-1]

    print(f"\n[+] MLP Total Training Time: {round(mlp_time_taken, 2)} seconds")
    print(f"[+] MLP Final Training Accuracy: {round(final_mlp_train_acc * 100, 2)}%")
    print(f"[+] MLP Final Test Accuracy:     {round(final_mlp_val_acc * 100, 2)}%")


def CNN_model():
    """
    Trains the Final Optimized CNN Model (Experiment #12) with Data Augmentation,
    Batch Normalization, Dropout, and Learning Rate Scheduling.
    """
    x_train_cnn = x_train / 255.0
    x_test_cnn = x_test / 255.0

    print("\n==========================================")
    print("Building and Training the Final CNN Model...")
    print("==========================================")
    
    # 1. Data Augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # 2. Final CNN Architecture (64 -> 128 -> 256 filters)
    inputs = layers.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Dense Classification Head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model_cnn = keras.Model(inputs=inputs, outputs=outputs, name="Final_CNN")
    model_cnn.summary()

    model_cnn.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 3. Learning Rate Scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        verbose=1
    )

    NUM_EPOCHS_CNN = 50
    cnn_start_time = time.time()
    
    history_cnn = model_cnn.fit(
        x_train_cnn,
        y_train,
        epochs=NUM_EPOCHS_CNN,
        validation_data=(x_test_cnn, y_test),
        callbacks=[lr_scheduler]
    )
    
    cnn_time_taken = time.time() - cnn_start_time
    final_train_acc = history_cnn.history['accuracy'][-1]
    final_val_acc = history_cnn.history['val_accuracy'][-1]
    
    print(f"\n[+] CNN Total Training Time: {round(cnn_time_taken, 2)} seconds")
    print(f"[+] Final CNN Training Accuracy: {round(final_train_acc * 100, 2)}%")
    print(f"[+] Final CNN Test Accuracy:     {round(final_val_acc * 100, 2)}%")


if __name__ == "__main__":
    MLP_model()
    CNN_model()