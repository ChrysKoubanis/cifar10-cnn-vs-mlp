import tensorflow as tf
from tensorflow import keras 
import numpy as np
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  #Loading the CIFAR-10 dataset

# print("Training data (shape):", x_train.shape)   
# print("Training labels (shape):", y_train.shape)
# print("Test data (shape):", x_test.shape)
# print("Test labels (shape):", y_test.shape)

# print("\nData type:", x_train.dtype)
# print("Min/Max pixel value:", np.min(x_train), "/", np.max(x_train))


#Now we need to preprocess the data (Flatten the images)
#32*32*3 = 3072 numbers per image ,so it will be a vector of size 3072
x_train_flat = x_train.reshape(x_train.shape[0],-1)    # x_train.shape[0] keeps the data the same (50000)
x_test_flat = x_test.reshape(x_test.shape[0],-1)    # -1 means flatten the dimensions (32*32*3) in to one dimension

#Now we flatten the labels from (50000,1) to (50000,)
y_train_flat =y_train.ravel() #Flattening the labels from 2D to 1D
y_test_flat = y_test.ravel()

#Now we divide the pixel values by 255 to normalize them between 0.0 and 1.0
x_train_flat=x_train_flat/255.0
x_test_flat=x_test_flat/255.0

#print("New flat images shape:", x_train_flat.shape)
#print("New flat labels shape:", y_train_flat.shape)


#Loading data and preprocessing is copy pasted from intermediate_assignment.py

#MLP Model
print("\n\nBuilding the MLP model...")
model_mlp = keras.Sequential([
    keras.layers.Input(shape=(3072,)), #Tells the model to expect a 1D vector with 3072 features (32*32*3)

    keras.layers.Dense(128, activation='relu'), #First hidden layer with 128 neurons and ReLU activation function
    keras.layers.Dense(64, activation='relu'), #Second hidden layer with 64 neurons and ReLU activation function
    keras.layers.Dense(10, activation='softmax') #Output layer with 10 neurons (one for each class) and softmax activation function (converts outputs into probabilities)
])

model_mlp.summary()  #Prints a summary of the model architecture