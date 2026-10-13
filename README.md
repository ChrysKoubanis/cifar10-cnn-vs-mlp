# CIFAR-10 Image Classification: MLP Baseline vs. Optimized CNN



An empirical benchmark evaluating spatial feature representations in Deep Learning by comparing Multi-Layer Perceptrons (MLPs) with regularized Convolutional Neural Networks (CNNs) on the CIFAR-10 image dataset.



---



## 📌 Project Overview



This repository demonstrates how architectural choices and regularization techniques directly impact an image classifier's ability to generalize to unseen data. 



While flattening 2D/3D image spatial structures causes dense Multi-Layer Perceptrons to hit an upper performance ceiling at ~49% accuracy, an iteratively optimized CNN leveraging Batch Normalization, Data Augmentation, Dropout, and Learning Rate Scheduling achieves 84.68% accuracy on the test set with minimal overfitting.



---



## 📊 Performance Summary



| Model | Model Architecture / Techniques | Training Time | Test Accuracy | Overfitting Gap | Key Observation |

| :--- | :--- | :--- | :--- | :--- | :--- |

| **NCC Baseline** | Nearest Class Center | ~2.6s | **27.74%** | — | Primitive baseline benchmark |

| **k-NN (k=1)** | Instance-based Non-parametric | ~13.0s | **35.39%** | — | High inference latency |

| **k-NN (k=3)** | Instance-based Non-parametric | ~13.0s | **33.03%** | — | Sensitive to background noise |

| **Baseline MLP** | 2 Hidden Layers (128 -> 64) | ~52.0s | **49.30%** | ~3.8% | Performance cap due to spatial flattening |

| **Baseline CNN** | 2 Conv Blocks (32 -> 64 filters) | ~75.0s | **69.73%** | ~20.1% | Severe overfitting without regularization |

| **Optimized CNN** | 3 Conv Blocks + BatchNorm + Augmentation + LR Scheduler | ~2052.0s | **84.68%** | ~3.4% | High test performance and strong generalization |



---

## 📈 Convergence & Learning Curves

![Training and Validation Performance](Figure_1%20CNN.png)[cite: 20]

---


## 🔬 Key Engineering Insights & Empirical Iteration



### 1. Spatial Structure Loss in MLPs

Flattening 32 x 32 x 3 image matrices into 1D vectors of 3,072 features strips localized spatial proximity (pixel neighborhood context). Scaling network width (adding hidden units) or depth only accelerated overfitting without surpassing the ~49% test accuracy barrier.



### 2. Systematic Regularization of the CNN

* **Addressing Severe Overfitting:** The baseline CNN reached 89.86% training accuracy but only 69.73% test accuracy (a 20% gap).

* **Dropout Optimization:** Adding a Dropout(0.5) layer before the output classification head closed the generalization gap down to ~1%.

* **Batch Normalization & Capacity Expansion:** Scaling to 3 Conv blocks (64 -> 128 -> 256 filters) with BatchNormalization stabilized activation distributions and sped up convergence.

* **Data Augmentation & Dynamic LR:** Applying random horizontal flips, rotations, and zooms alongside Keras's ReduceLROnPlateau callback allowed the model to reach its peak test performance of 84.68% over 50 training epochs.



---



## 🏗️ Final CNN Architecture


Input (32x32x3 RGB Image)
  │
  ├── Data Augmentation (Random Flip, Rotation, Zoom)
  │
  ├── Conv2D (64 filters, 3x3) ──► BatchNormalization ──► ReLU ──► MaxPooling2D (2x2)
  ├── Conv2D (128 filters, 3x3) ──► BatchNormalization ──► ReLU ──► MaxPooling2D (2x2)
  ├── Conv2D (256 filters, 3x3) ──► BatchNormalization ──► ReLU ──► MaxPooling2D (2x2)
  │
  ├── Flatten
  ├── Dense (128 units, ReLU)
  ├── Dropout (0.5)
  └── Dense (10 units, Softmax Output)


## 🛠️ Setup & Execution



### Prerequisites

* Python 3.9+

* TensorFlow 2.x / Keras



### Running the Code

1. Clone the repository:

   ```bash

   git clone git clone https://github.com/ChrysKoubanis/cifar10-cnn-vs-mlp.git
   cd cifar10-cnn-vs-mlp

   ```



2. Execute the evaluation script:

   ```bash

   python main_assignment.py

   ```



---



## 🧰 Technologies & Tools



* **Core Framework:** Python, TensorFlow / Keras

* **Data Processing & Analytics:** NumPy, Matplotlib