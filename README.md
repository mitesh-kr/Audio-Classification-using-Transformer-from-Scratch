# Environmental Sound Classification using Transformers from Scratch

This repository contains an implementation of a Transformer network and a CNN-based model for classifying environmental audio recordings into 10 different classes. The models are implemented in PyTorch.

## 📌 Objective

The goal of this project is to implement and compare two architectures:

1. **CNN-based Model**: A 1D convolutional neural network (CNN) for feature extraction followed by a fully connected classification layer.
2. **Transformer-based Model**: A CNN feature extractor followed by a Transformer encoder with a multi-head self-attention mechanism.

Both architectures are evaluated for accuracy, loss, and other key performance metrics.

---

## 🗂 Dataset

- The dataset consists of **400 environmental audio recordings** categorized into **10 different classes**.
- Preprocessing is handled using a custom data loader.
- [Download Dataset](https://iitjacin-my.sharepoint.com/personal/mishra_10_iitj_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmishra%5F10%5Fiitj%5Fac%5Fin%2FDocuments%2FAudio%5FAssignment%5FDL%2FArchive%2Ezip&parent=%2Fpersonal%2Fmishra%5F10%5Fiitj%5Fac%5Fin%2FDocuments%2FAudio%5FAssignment%5FDL&ga=1)


---

## 🏗 Network Architectures

### 🔹 **Architecture 1: CNN-based Model**
- Uses **1D-convolution** for feature extraction.
- Comprises at least **three convolutional layers**.
- A **fully connected layer** is used for classification.

### 🔹 **Architecture 2: Transformer-based Model**
- Uses the same **1D-CNN** feature extractor as Architecture 1.
- A **Transformer encoder** is implemented from scratch, including:
  - Multi-head self-attention mechanism (**heads = 1, 2, 4**)
  - Addition of `<cls>` token.
- A **MLP head** is used for final classification.

---

## 📌 Tasks & Methodology

✔ **Training Configuration**
- Models are trained for **100 epochs**.
- Training logs (accuracy & loss) are plotted using **Weights & Biases (WandB)**.

✔ **Validation Strategy**
- **4-fold cross-validation** is performed.

✔ **Evaluation Metrics**
- **Accuracy**
- **Confusion Matrix**
- **F1-score**
- **AUC-ROC Curve**

✔ **Parameter Analysis**
- Total **trainable and non-trainable parameters** are reported.
- **Hyperparameter tuning** is performed to find the best configuration.

---

## ⚡ Setup & Installation

### 🔹 Clone the Repository

```bash
git clone https://github.com/mitesh-kr/Environmental-Audio-Classification.git
cd Environmental-Audio-Classification
