# Quantum Network Intrusion Detection System

## Overview

The Quantum Network Intrusion Detection System (QNIDS) is a cybersecurity project that detects malicious network traffic using both Classical Support Vector Machines (SVM) and Quantum Support Vector Machines (QSVM).

The project uses the NSL-KDD dataset and applies preprocessing, feature engineering, dimensionality reduction, and machine learning techniques to classify network attacks.

---

## Objectives

* Detect cyberattacks in network traffic.
* Compare Classical and Quantum Machine Learning approaches.
* Demonstrate the applicability of Quantum Computing in cybersecurity.

---

## Dataset

* NSL-KDD Dataset

### Attack Categories

1. Normal
2. DoS (Denial of Service)
3. Probe
4. R2L (Remote to Local)
5. U2R (User to Root)

---

## Technologies Used

* Python
* Scikit-learn
* Qiskit
* NumPy
* Pandas
* Matplotlib
* Seaborn

---

## Project Structure

```
dataset/
preprocessing/
models/
alerts/
results/
notebooks/

main.py
README.md
requirements.txt
```

---

## Workflow

1. Dataset Loading
2. Attack Mapping
3. Feature Encoding
4. Feature Scaling
5. Train-Test Split
6. PCA Feature Reduction
7. Classical SVM Classification
8. Quantum SVM Classification
9. Alert Generation
10. Result Visualization

---

## Results

### Classical SVM Accuracy

96.75%

### Quantum SVM Accuracy

85.00%

### PCA Variance Retained

92.48%

---

## Future Enhancements

* Hybrid Classical + Quantum IDS
* Real-time intrusion monitoring
* Streamlit dashboard
* CICIDS2017 dataset support
* Email-based alerts
