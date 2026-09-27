# Comparative Analysis of ECG Preprocessing Methods for Deep Learning-Based Arrhythmia Detection Using CNN-LSTM

This repository contains the implementation for benchmarking and comparatively analyzing the impact of various electrocardiogram (ECG) preprocessing pipelines on deep learning-based arrhythmia detection using a hybrid **CNN-LSTM** architecture.

---

## 📌 Project Overview
Arrhythmia detection from raw ECG signals is often degraded by baseline wander, powerline interference, and muscle artifacts. This research systematically investigates how different stages of signal preprocessing (filtering, normalization, and downsampling) affect the classification performance and inference efficiency of a cost-sensitive CNN-LSTM model on the **MIT-BIH Arrhythmia Database**.

---

## 🔬 Experimental Scenarios

The study evaluates 7 hierarchical preprocessing scenarios:

| Scenario | Filtering Method | Normalization | Sampling Rate | Purpose |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Raw Signal (None) | None | 360 Hz | Baseline performance evaluation |
| **2** | Butterworth Bandpass (0.5 – 45 Hz) | None | 360 Hz | Conventional high/low-frequency noise removal |
| **3** | DWT Symlet-10 (`sym10`) | None | 360 Hz | Wavelet baseline wander & noise removal preserving QRS morphology |
| **4** | Best Filter (from 2 or 3) | Min-Max Scaling ($[-1, 1]$) | 360 Hz | Amplitude range bounding |
| **5** | Best Filter | Z-Score ($\mu=0, \sigma=1$) | 360 Hz | Distributional standard scaling |
| **6** | Best Filter + Best Normalization | Resampled | 100 Hz | Memory efficiency & latency test |
| **7** | Best Filter + Best Normalization | Resampled | 50 Hz | Lower-bound sampling rate tolerance test |

---

## 🫀 Dataset & Annotation Protocol
- **Dataset:** [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/) (PhysioNet).
- **Lead:** Modified Limb Lead II (MLII).
- **Target Classes (AAMI EC57 Standard):**
  - **N:** Normal & bundle branch blocks
  - **S:** Supraventricular ectopic beats
  - **V:** Ventricular ectopic beats
  - **F:** Fusion beats
  - **Q:** Unknown / Paced beats
- **Splitting Strategy:** Patient-independent data split (inter-patient protocol) to prevent data leakage between training and testing sets.

---

## 🧠 Model Architecture & Evaluation
- **Model:** 1D Convolutional Neural Network followed by Long Short-Term Memory (CNN-LSTM) hybrid network.
- **Handling Class Imbalance:** Cost-sensitive weighted loss function.
- **Evaluation Metrics:** Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, and Inference Latency.

---

## 🚀 Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/danielsetiawn/ECG-analysis-for-arrhythmia-detection-with-deep-learning.git
   cd ECG-analysis-for-arrhythmia-detection-with-deep-learning
   ```

2. **Install dependencies:**
   ```bash
   pip install wfdb pywavelets scipy numpy pandas scikit-learn tensorflow matplotlib
   ```

3. **Run experiments:**
   Open and execute `main.ipynb`.
