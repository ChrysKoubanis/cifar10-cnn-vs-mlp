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
model_mlp = keras.Sequential([  #Sequential model means layers are stacked one after another
    keras.layers.Input(shape=(3072,)), #Tells the model to expect a 1D vector with 3072 features (32*32*3)
    keras.layers.Dense(128, activation="relu"), #First hidden layer with 128 neurons and ReLU activation function
    keras.layers.Dense(64, activation="relu"), #Second hidden layer with 64 neurons and ReLU activation function
    keras.layers.Dense(10, activation="softmax") #Output layer with 10 neurons (one for each class) and softmax activation function (converts outputs into probabilities)
])

model_mlp.summary()  #Prints a summary of the model architecture  
#In output Shape, None means batch size can be any number ,which means that it can accept any number of images at once
#Parameters are the number of weights and biases that the model will learn during training,
# in dense_2 its 64 inputs per neuron *10 neurons (this layers neurons) +10 biases (1 bias knob per neuron) =650
# in dense_1 its 128 inputs per neuron *64 neurons +64 biases =8256
# in dense its 3072 inputs per neuron *128 neurons +128 biases = 393344

print("\nCompiling the MLP model...")

model_mlp.compile(optimizer="adam",  #Adam optimizer is an efficient optimization algorithm that adjusts learning rates during training
                    loss="sparse_categorical_crossentropy", #Loss function for multi-class classification problems (like our problem) where labels are integers
                    metrics=["accuracy"])  #is to report accuracy as it trains


print("\nTraining the MLP model...")
NUM_EPOCHS = 10  #Number of times the model will go through the entire training dataset
mlp_start_time = time.time()
model_mlp.fit(x_train_flat, y_train_flat, epochs=NUM_EPOCHS, validation_data=(x_test_flat, y_test_flat))  
#Training the model with training data and validating with test data after each epoch
mlp_time_taken = time.time() - mlp_start_time
print("\nMLP Total Training Time:", mlp_time_taken, "seconds")

#First attempt 10 epochs, 2 hidden layers, one with 128 and one with 64 neurons took around 18 seconds ,
#the training (46.3%) and test (45.9%) accuracies are very close.
#The model is underfitting ,we will increase the epochs.

#Second attempt 20 epochs, 2 hidden layers, one with 128 and one with 64 neurons took around 37 seconds ,
#the training (50.3%) and test (48.3%) accuracies have started to diverge slightly.
#That means that the model has started overfitting

#Third attempt 50 epochs, 2 hidden layers, one with 128 and one with 64 neurons took around 105 seconds,
#the training (54%) and test (49%) accuracies have a 5% gap,
#training accuracy has improved but there is a small difference in test accuracy, that means that we have hit a plateau and the model is overfitting.

#I want to play with the model more, i will try increasing the number of neurons and layers.
#Fourth attempt 20 epochs 2 hidden layers, one with 256 and one with 128 neurons took around 52 seconds,
#the training (53.1%) and test (49.3%) accuracies still a big gap, not much improvement, the model is OVERFITTING.

#Fifth attempt 20 epochs 3 hidden layers, one with 256, one with 128 and one with 64 neurons took around 54 seconds,
# the training (53.9%) and test (48.8%) accuracies ,worse than before, the model is overfitting more.


# After multiple experiments (longer training, wider layers, deeper layers),
# the MLP model consistently hits a "performance wall" at around 49-50% test accuracy.
#
# Any attempt to add more power to the model just results in OVERFITTING
# (the training accuracy goes up, but the test accuracy does not).

# Now lets see some situations where the model isnt overfitting:

# First attempt 10 epochs 2 hidden layers, one with 64 and one with 32 neurons took around 16 seconds,
# the training (44.6%) and test (44.5%) accuracies ,less accurate than before, but isnt overfitting.

# Second attempt 10 epochs 1 hidden layer with 256 neurons took around 24 seconds,
# the training (46.5%) and test (45.7%) accuracies ,only one layer, but better than first attempt (because there are more neurons).

# Third attempt 10 epochs 1 hidden layer with 512 neurons took around 51 seconds,
# the training (46.9%) and test (46.3%) accuracies , we have hit the plateau of only having one layer.

# Forth attempt 10 epochs 2 hidden layers, one with 50, one with 100 neurons took around 23 seconds,
# the training (47.5%) and test (46.8%) accuracies , we have hit the plateau of having two layers.

#Conclusion:
# The MLP model is underfitting on every situation because of the models architecture.
# If we make the model bigger (more layers or neurons or epochs), it overfits.
# This strongly suggests the model is limited by the INPUT DATA:
# The flattened 3072-D vector has lost all the spatial structure
# (what pixels are "next to" each other) and is confused by background noise.
# To get a better score, we MUST use a model that understands
# 2D image structure: a Convolutional Neural Network (CNN).