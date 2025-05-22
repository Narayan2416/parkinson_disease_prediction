import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np

# ======================= Style Section =======================
st.set_page_config(page_title="Parkinson's Classifier", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f7f7f7;
    }
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #fff3cd;
        padding: 20px;
        border-left: 6px solid #ffcc00;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 18px;
    }
    .stNumberInput label {
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# ======================= Model Loading =======================
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_pca():
    return joblib.load("pca.pkl")

@st.cache_resource
def load_model():
    model = ANN()
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Define the model class
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(pca.n_components_, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.BatchNorm1d(75),
            nn.ReLU(),
            nn.Linear(75, 25),
            nn.BatchNorm1d(25),
            nn.ReLU(),
            nn.Linear(25, 5),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ======================= Load Assets =======================
scaler = load_scaler()
pca = load_pca()
model = load_model()
feature_names = scaler.feature_names_in_.tolist()

@st.cache_data
def get_pca_labels(_pca, feature_names, top_n=3):
    labels = []
    for i, component in enumerate(_pca.components_):
        top_indices = np.argsort(np.abs(component))[::-1][:top_n]
        top_features = [feature_names[idx] for idx in top_indices]
        label = f"PCA {i+1} ({', '.join(top_features)})"
        labels.append(label)
    return labels

pca_labels = get_pca_labels(pca, feature_names, top_n=3)

# ======================= App UI =======================
st.markdown("<div class='main-title'>🧠 Parkinson's Disease Classifier</div>", unsafe_allow_html=True)
st.write("Enter PCA-reduced input features to predict the likelihood of Parkinson's symptoms.")

# Group input fields into expandable rows with columns
pca_input_features = []
st.subheader("Enter values for PCA-reduced features:")

with st.expander("🔧 PCA Feature Inputs", expanded=True):
    cols = st.columns(3)  # 3 columns layout
    for i, label in enumerate(pca_labels):
        col = cols[i % 3]
        with col:
            value = st.number_input(label, value=0.0, key=f"pca_{i}")
            pca_input_features.append(value)

# ======================= Prediction =======================
if st.button("🔍 Predict"):
    input_pca = np.array(pca_input_features).reshape(1, -1)
    input_tensor = torch.tensor(input_pca, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()
        label = "Symptoms of Parkinson's" if prediction > 0.5 else "a Healthy Body"
        confidence = (prediction * 100) if prediction > 0.5 else ((1 - prediction) * 100)
        color = "#ff4b4b" if prediction > 0.5 else "#28a745"

        st.markdown(f"""
            <div class="prediction-box">
                <b>Prediction:</b> <span style="color:{color}">{label}</span><br>
                <b>Confidence:</b> {confidence:.2f}%
            </div>
        """, unsafe_allow_html=True)
