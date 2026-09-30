# Out-of-Distribution (OOD) Detection using Mahalanobis Distance on Animals with Attributes 2 (AwA2)

An advanced computer vision pipeline built in PyTorch to detect **Out-of-Distribution (OOD) anomalies** and unknown inputs using **Mahalanobis Distance-based Confidence Scores** and **Multi-Layer Feature Analysis** on a ResNet-18 architecture.

---

## 🚀 Project Overview

Standard deep learning classifiers (like Softmax) are notoriously overconfident when presented with inputs they have never seen before. This project implements a robust OOD detection framework using **feature-space Mahalanobis distance metrics** to separate in-distribution (ID) test samples from out-of-distribution anomaly sets.

* **In-Distribution (ID) Dataset:** [Animals with Attributes 2 (AwA2)](https://www.google.com/search?q=https://cvml.ist.ac.at/AwA2/) (Animal classes)
* **Out-of-Distribution (OOD) Dataset:** SVHN (Street View House Numbers) / Custom uploaded anomalies
* **Backbone Architecture:** ResNet-18 (Pre-trained on ImageNet, fine-tuned on AwA2)
* **Core Anomaly Metric:** Mahalanobis Distance relative to class conditional Gaussian distributions with covariance regularization.

---

## 🛠️ Key Technical Features

1. **Dual-Output Feature Extraction:** Isolates intermediate representations and latent vectors before the final classification head.
2. **Covariance Regularization:** Employs an epsilon-smoothed identity matrix ($\epsilon = 0.01$) to handle high-dimensional collinearity and stabilize matrix inversion.
3. **Multi-Layer Hook Profiling:** Registers forward hooks across multiple residual blocks (`layer1` through `layer4`) to evaluate multi-scale anomaly detection (capturing texture vs. high-level semantic geometry).
4. **Comprehensive Evaluation Suite:** Measures detection performance using **AUROC (Area Under the ROC Curve)**, comparing standard Softmax confidence against Mahalanobis distance rejection.
5. **Dimensionality Reduction & Visualization:** Utilizes PCA to project high-dimensional feature spaces ($512$-D) down to 3D for intuitive latent cluster visualization.

---

## 📦 Project Structure & Workflow

The codebase is structured into sequential Google Colab blocks:

* **Cell 1:** Data acquisition (auto-downloads AwA2 and SVHN) and standard preprocessing transforms (`224x224` resize, normalization).
* **Cell 2:** Model definition (`DualOutputResNet`) and 1-epoch quick fine-tuning loop.
* **Cell 3:** Feature extraction, class centroid estimation, and regularized tied covariance calculation.
* **Cell 4 & 5:** OOD evaluation scripts calculating AUROC scores and plotting 3D latent PCA clusters.
* **Cell 6 & 7:** Multi-layer hook calibration and live anomaly inference testing.

---

## 💻 Getting Started & Installation

### Prerequisites

Make sure you are running this in a Python/PyTorch environment (Google Colab is recommended for GPU acceleration):

```bash
pip install torch torchvision scikit-learn matplotlib tqdm

```

### Running the Pipeline

1. Execute the setup cells to pull down the **AwA2** dataset.
2. Let the model complete its fine-tuning phase on the training split.
3. Run the evaluation cell to generate the **AUROC score comparison** between standard Softmax and Mahalanobis distance.
4. Run the **Live OOD Tester** cell to upload any custom image and check if the system accepts it as a known animal or rejects it as an anomaly.

---

## 📊 Results Summary

* **Standard Softmax Baseline:** Suffers from overconfidence on unknown inputs.
* **Mahalanobis Distance Method:** Significantly outperforms baseline by measuring exact statistical deviation from learned class hyperplanes, achieving superior separation and higher **AUROC**.
