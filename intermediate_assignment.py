import tensorflow as tf
import numpy as np

print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Training data (shape):", x_train.shape)
print("Training labels (shape):", y_train.shape)
print("Test data (shape):", x_test.shape)
print("Test labels (shape):", y_test.shape)

print("\nData type:", x_train.dtype)
print("Min/Max pixel value:", np.min(x_train), "/", np.max(x_train))