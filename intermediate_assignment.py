import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# print("Training data (shape):", x_train.shape)   
# print("Training labels (shape):", y_train.shape)
# print("Test data (shape):", x_test.shape)
# print("Test labels (shape):", y_test.shape)

# print("\nData type:", x_train.dtype)
# print("Min/Max pixel value:", np.min(x_train), "/", np.max(x_train))

#Now we need to preprocess the data (Flatten the images)
#32*32*3 = 3072 numbers per image ,so it will be a vector of size 3072
x_train_flat = x_train.reshape(x_train.shape[0],-1)
x_test_flat = x_test.reshape(x_test.shape[0],-1)

#Now we flatten the labels from (50000,1) to (50000,
y_train_flat =y_train.ravel()
y_test_flat = y_test.ravel()

#Now we devide the pixel values by 255 to normalize them between 0.0 and 1.0
x_train_flat=x_train_flat /255.0
x_test_flat=x_test_flat /255.0

print("New flat train images shape:", x_train_flat.shape)
print("New flat train labels shape:", y_train_flat.shape)