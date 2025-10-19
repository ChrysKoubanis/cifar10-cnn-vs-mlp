# Project: Image Classification with Neural Networks
This is a project for the "Neural Networks – Deep Learning" course. The goal is to build and evaluate models for classifying images from the **CIFAR-10** dataset.

## 🎯 Project Goal

1.  **Baseline Models:** Implement and test k-NN (k=1, k=3) and Nearest Class Centroid (NCC) classifiers.
2.  **Main Assignment:** Build a Feedforward Neural Network (MLP or CNN) to solve the same problem.
3.  **Analysis:** Compare the Neural Network's performance against the baseline models.

---

## 🛠️ Part 1: Intermediate Assignment (Baseline Models)

In this part, we established baseline performance using classic machine learning classifiers. The data was flattened into 3072-dimension vectors and normalized.

The models were trained and tested on the **full CIFAR-10 dataset** (50,000 train, 10,000 test images).

### Baseline Results

| Classifier | Parameters | Accuracy | Time (seconds) |
| :--- | :--- | :--- | :--- |
| **Nearest Class Centroid (NCC)** | - | 27.74% | ~2.75 sec |
| **k-Nearest Neighbors (k-NN)** | `k=1` | **35.39%** | ~11.12 sec |
| **k-Nearest Neighbors (k-NN)** | `k=3` | 33.03% | ~10.88 sec |

**Conclusion:** The best-performing baseline model is **k-NN (k=1)**, with **35.39%** accuracy.

---

## 🧠 Part 2: Main Assignment (Neural Network)

*(This section will be updated after the main assignment is complete.)*