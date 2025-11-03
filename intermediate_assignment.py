import tensorflow as tf
import numpy as np

from sklearn.neighbors import NearestCentroid , KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()  #Loading the CIFAR-10 dataset

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

#NCC
print("\n\nTesting the NCC model...")
ncc_start_time = time.time() 
ncc_model =NearestCentroid() # The Nearest Centroid Classifier model
ncc_model.fit(x_train_flat, y_train_flat) #Fitting the model with training data
ncc_preds=ncc_model.predict(x_test_flat) #Predicting the labels for test data

ncc_accuracy=accuracy_score(y_test_flat, ncc_preds) #Calculating accuracy
ncc_time_taken = time.time() -ncc_start_time #Calculating time taken

print("NCC Accuracy:", ncc_accuracy *100, "%")   # Accuracy is 27.74%
print("NCC Total Time:", ncc_time_taken, "seconds")   # Total time taken is around 2.4-2.7 seconds

#k-NN for k=1
print("\nTesting the k-NN model with k=1...")
knn1_start_time=time.time()

knn1_model=KNeighborsClassifier(n_neighbors=1,n_jobs=-1)  #small difference between n_jobs=1 (one CPU core) and n_jobs=-1 (all available CPU cores),not what i expected
knn1_model.fit(x_train_flat,y_train_flat)
knn1_preds=knn1_model.predict(x_test_flat)

knn1_accuracy =accuracy_score(y_test_flat, knn1_preds)
knn1_time_taken =time.time() -knn1_start_time

print("k-NN (k=1) Accuracy:", knn1_accuracy* 100,"%")  # Accuracy is 35.39%
print("k-NN (k=1) Total Time:", knn1_time_taken, "seconds")  # Total time taken is around 13 seconds

#k-NN for k=3
print("\nTesting the k-NN model with k=3...")
knn3_start_time=time.time()

knn3_model=KNeighborsClassifier(n_neighbors=3,n_jobs=-1) 
knn3_model.fit(x_train_flat,y_train_flat)
knn3_preds=knn3_model.predict(x_test_flat)

knn3_accuracy =accuracy_score(y_test_flat, knn3_preds)
knn3_time_taken =time.time() -knn3_start_time

print("k-NN (k=3) Accuracy:", knn3_accuracy*100,"%") # Accuracy is 33.03 %
print("k-NN (k=3) Total Time:", knn3_time_taken,"seconds") # Total time taken is around 13 seconds


# Now we will generate graphs to compare the results of the three models
print("\n\n--- Generating Results Graphs ---")
models = ["NCC", "k-NN (k=1)", "k-NN (k=3)"]
accuracies = [ncc_accuracy * 100, knn1_accuracy * 100, knn3_accuracy * 100] # *100 to convert to percentage
times = [ncc_time_taken, knn1_time_taken, knn3_time_taken]


plt.figure(figsize=(12, 6)) # One figure for both plots

# Plot 1: Accuracy Comparison
plt.subplot(1,2,1)
plt.bar(models, accuracies, color=["blue","red","green"])
plt.title("Classifier Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0,100)

# Add the accuracy percentage on top of each bar
for i in range(len(models)):
    plt.text(i, accuracies[i] + 1, f"{accuracies[i]:.2f}%", ha="center")

    # Plot 2: Time Comparison
plt.subplot(1, 2, 2)
plt.bar(models, times,color=["blue","red","green"])
plt.title("Classifier Time Comparison")
plt.ylabel("Time (seconds)")

# Add the time on top of each bar
for i in range(len(models)):
    plt.text(i, times[i] + 0.2, f"{times[i]:.2f}s", ha="center")

plt.suptitle("Intermediate Assignment Results (Full CIFAR-10)")
plt.tight_layout()
plt.show()