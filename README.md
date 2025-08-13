# 🧠 Parkinson's Disease Prediction App

This repository contains a deep learning–based web application for **detecting Parkinson’s Disease from EEG signal** data. The EEG features are reduced using Principal Component Analysis (PCA) and classified with an **Artificial Neural Network (ANN)** implemented in TensorFlow. The application, built with Streamlit, offers an intuitive interface for real-time and accessible health assessment.

🔗 **Live Demo:** [https://parkinson-disease-prediction-887j.onrender.com](https://parkinson-disease-prediction-887j.onrender.com)

---

## 🚀 Features

- 🎯 **Binary Classification** – Predicts Parkinson’s symptoms or healthy condition.
- ⚡ **Real-Time Predictions** – Instant result based on user input features.
- 🧠 **Deep Learning Model** – Neural network with batch normalization and activations.
- 📉 **PCA Integration** – Uses principal component analysis to reduce dimensionality.
- 🎨 **Streamlit UI** – Clean, interactive, and styled interface.

---

## 🧰 Tech Stack

| Technology     | Purpose                          |
|----------------|----------------------------------|
| Python 3.8+     | Core language                    |
| Streamlit       | Web UI and deployment            |
| TensorFlow      | Model training and inference     |
| Joblib          | Scaler and PCA persistence       |
| NumPy           | Data handling                    |

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Narayan2416/parkinson_disease_prediction.git
cd parkinsons_disease_prediction
pip install -r requirements.txt

## Run the app by
streamlit run main.py
