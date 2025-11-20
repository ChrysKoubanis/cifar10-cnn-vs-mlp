import tensorflow as tf
from tensorflow import keras 
#from tensorflow.keras import layers
import numpy as np
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  #Loading the CIFAR-10 dataset

# print("Training data (shape):", x_train.shape)   
# print("Training labels (shape):", y_train.shape)
# print("Test data (shape):", x_test.shape)
# print("Test labels (shape):", y_test.shape)

# print("\nData type:", x_train.dtype)
# print("Min/Max pixel value:", np.min(x_train), "/", np.max(x_train))

def MLP_model():
    
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
    #the training (50.3%) and test (48.3%) accuracies have started to diverge slightly.Still underfitting
    
    #Third attempt 50 epochs, 2 hidden layers, one with 128 and one with 64 neurons took around 105 seconds,
    #the training (54%) and test (49%) accuracies have a 5% gap,
    #training accuracy has improved but there is a small difference in test accuracy, that means that we have hit a plateau and the model is overfitting.Still underfitting

    #I want to play with the model more, i will try increasing the number of neurons and layers.
    #Fourth attempt 20 epochs 2 hidden layers, one with 256 and one with 128 neurons took around 52 seconds,
    #the training (53.1%) and test (49.3%) accuracies still a big gap, not much improvement, the model is overfitting.Still underfitting

    #Fifth attempt 20 epochs 3 hidden layers, one with 256, one with 128 and one with 64 neurons took around 54 seconds,
    # the training (53.9%) and test (48.8%) accuracies ,worse than before, the model is overfitting more.Still underfitting


    # After multiple experiments (longer training, wider layers, deeper layers),
    # the MLP model consistently hits a "performance wall" at around 49-50% test accuracy.That means the model is underfitting.
    #
    # Any attempt to add more power to the model just results in overfitting
    # (the training accuracy goes up, but the test accuracy does not).

    # Now lets see some situations where we make the model smaller 
    # First attempt 10 epochs 2 hidden layers, one with 64 and one with 32 neurons took around 16 seconds,
    # the training (44.6%) and test (44.5%) accuracies ,less accurate than before, but isnt overfitting.

    # Second attempt 10 epochs 1 hidden layer with 256 neurons took around 24 seconds,
    # the training (46.5%) and test (45.7%) accuracies ,only one layer, but better than first attempt (because there are more neurons).

    # Third attempt 10 epochs 1 hidden layer with 512 neurons took around 51 seconds,
    # the training (46.9%) and test (46.3%) accuracies , we have hit the plateau of only having one layer.

    #Conclusion:
    # The MLP model is underfitting on every situation because of the models architecture.
    # If we make the model bigger (more layers or neurons or epochs), it overfits.
    # This strongly suggests the model is limited by the INPUT DATA:
    # The flattened 3072-D vector has lost all the spatial structure
    # (what pixels are "next to" each other) and is confused by background noise.
    # To get a better score, we MUST use a model that understands
    # 2D image structure: a Convolutional Neural Network (CNN).


#Lets start building the CNN model now.I want to focus more on the CNN model, so i will do more experiments.
#CNN Model, this is the "smart" model that will use the original 3D images
#We already have the data loaded in (x_train, y_train), (x_test, y_test), but lets create new variables for clarity
def CNN_model():
    x_train_cnn=x_train/255.0 #normalize the values between 0.0 and 1.0
    x_test_cnn=x_test/255.0
    #print("CNN data (x_train_cnn) shape:", x_train_cnn.shape)   #(50000, 32, 32, 3)
    #print("CNN labels (y_train) shape:", y_train.shape)         #(50000, 1)
    print("\n\nBuilding the CNN model...")
    model_cnn = keras.Sequential([
        keras.layers.Input(shape=(32,32,3)) , #Input layer for our 3D images of size 32x32 with 3 colors
        
    
    # 1. First "Conv2D" Layer:
    #This is the "Feature Extractor" ,the "eyes"
    #It will learn 32 different "filters" (32 "magnifying glasses").
    # (3,3)= Each filter is a 3x3 pixel grid.
    keras.layers.Conv2D(32, (3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    
    # 2. "MaxPooling2D" Layer:
    #This shrinks (cuts height and width in half) the image to make it more manageable.
    #It looks at 2x2 blocks of pixels and keeps only the biggest number ,to focus on the most important features.
    keras.layers.MaxPooling2D((2, 2)),
    
    # 3. Second "Conv2D" Layer:
    #We do it again. This layer learns to *combine* the simple features (like edges) from the first layer 
    #into more complex features (like corners or curves).
    keras.layers.Conv2D(64, (3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    
    # 4. Second "MaxPooling2D" Layer:
    #Shrinks it again.
    keras.layers.MaxPooling2D((2, 2)),

    # 5. Third "Conv2D" Layer:
    # It wll find more complex shapes (parts of objects) (Image size: 8x8)
    # We double the filters again (from 64 to 128) to increase power.
    # I wont use MAxpool because the image is already small (8x8).
    keras.layers.Conv2D(128, (3, 3),  padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    
    # 6. "Flatten" Layer:
    #We're done finding features.Now we need to transition the data to a 1D "list" format
    #that a Dense layer can understand.
    keras.layers.Flatten(),
    
    # 7. "Dense" Layer:
    #This is a "fully-connected" layer
    #It's the "brain" that will look at all the features the "eyes" found and start to make a decision.
    keras.layers.Dense(64,),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.Dropout(0.5), # Dropout layer to reduce overfitting by randomly turning off 50% of neurons during training
    
    # 8. Output Layer:
    #Softmax converts the scores into probabilities.
    keras.layers.Dense(10, activation='softmax')])

    model_cnn.summary()

    print("\nCompiling the CNN model...")
    model_cnn.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    
    print("\nTraining the CNN model...")
    NUM_EPOCHS_CNN = 15
    
    cnn_start_time = time.time()
    
    history_cnn = model_cnn.fit(
        x_train_cnn,
        y_train,
        epochs=NUM_EPOCHS_CNN,
        validation_data=(x_test_cnn, y_test)
    )
    
    cnn_time_taken=time.time()-cnn_start_time
    print("\nCNN Total Training Time: ",round(cnn_time_taken,2),"seconds")
    
    # Let's print the final accuracy
    # The 'history_cnn' object stores all the results
    final_train_acc = history_cnn.history['accuracy'][-1]
    final_val_acc = history_cnn.history['val_accuracy'][-1]
    
    print("Final CNN Training Accuracy:", round(final_train_acc * 100,2),"%")
    print("Final CNN Test (Validation) Accuracy: ",round(final_val_acc * 100,2),"%")

#Starting architecure is: applying 32 filters, shrinking the image twice with max pooling, applying 64 filters, shrinking the image twice with max pooling,
#flattening the image, one dense layer with 64 neurons and output layer for the 10 classes.I havent twicked the architecture yet. Epochs are set to 15.


#First attempt took around 75 seconds, with training accuracy of 89.86% and test accuracy of 69.73% , there is a gap of 20% ,thats considered significant overfitting.
#The model is too powerful and is memorizing the training data instead of generalizing well to unseen data.
#Adding more neurons or layers will only make the overfitting worse.So i will try to reduce overfitting with a regularization technique called dropout layer.
#The dropout layer randomly "turns off" a percentage neurons during training, forcing the model to learn the general patterns in the data rather than memorizing the noise.

#Second attempt: I added a dropout layer after the 64 neuron dense layer with a dropout rate of 0.3 (30% of neurons will be turned off during training).
#Second attempt took around 83 seconds, with training accuracy of 75.46% and test accuracy of 70.75% , the gap is now around 5%, which is a big improvement, but the model is underfitting its potential.
#I want to first try to tweak the dropout rate , then see if more epochs will help with underfitting and then make the model more powerful.

#Third attempt: I changed the dropout rate from 0.3 to 0.2
#Third attempt took around 83 seconds, with training accuracy of 77.06% and test accuracy of 70.05% , the gap is around 7%.
#Fourth attempt: I changed the dropout rate from 0.3 to 0.4
#Fourth attempt took around 92 seconds, with training accuracy of 70.64% and test accuracy of 70.18% , the gap is around 0.5%.
#Since i got better results with 0.4 dropout rate, i will try 0.5 dropout rate next, but i dont expect it to work miracles.
#Fifth attempt: I changed the dropout rate from 0.4 to 0.5
#Fifth attempt took around  88 seconds, with training accuracy of 70.80% and test accuracy of 71.54% , the gap is around 1%.
#Very interestingly, the test accuracy is higher than training accuracy, this happened because the dropout layer is only active during training,
#so during testing all neurons are active, which can lead to slightly better performance on unseen data.
#I want to find the best dropout rate, so i will try 0.6 next.
#Sixth attempt: I changed the dropout rate from 0.5 to 0.6
#Sixth attempt took around 80 seconds, with training accuracy of 63.47% and test accuracy of 68.21% , the gap is around 5%.
#The dropout rate of 0.6 is too high, as expected, so i will stick with 0.5 ,since it gave the best results so far.
#I now want to increase the number of epochs to see if the model can learn more without overfitting.
#Seventh attempt: I increased the number of epochs from 15 to 30, keeping the dropout rate at 0.5
#Seventh attempt took around 167 seconds, with training accuracy of 75.97% and test accuracy of 70.70% , the gap is around 5%.
#We can see that training for longer hurt the model, because we gave the model too much time and it started to memorize the training data again.
#That means that at 15 epochs the model was at its best performance, underfitting but healthy gap.

#I will return to 15 epochs and try to make the model more powerful, while keeping the dropout rate at 0.5 to control overfitting.
#To improve, we need a more powerful model that can learn more complex features.
#Eighth attempt: I added a 3rd Conv2D layer with 128 filters to make the model more powerful because it was underfitting its potential.
#Eighth attempt took around 144 seconds, with training accuracy of 81.22% and test accuracy of 74.08%. Test accuracy improved, but the gap is around 7%, so the model is overfitting again.
#I will now add a batch normalization layer after each Conv2D layer to stabilize and speed up training.
#Ninth attempt: I added batch normalization layers after each Conv2D layer.

CNN_model()