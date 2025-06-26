import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Input,Dropout,ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# ======================= Style Section =======================
st.set_page_config(page_title="Parkinson's Classifier", layout="wide")

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

# ======================= Model Architecture Definition =======================
def build_model(input_shape):
    model=Sequential([
        Input(shape=(input_shape,)),
        Dense(150),
        BatchNormalization(),
        ReLU(),
        Dense(75),
        BatchNormalization(),
        ReLU(),
        Dense(25),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.3),
        Dense(5),
        LeakyReLU(),
        BatchNormalization(),
        Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ======================= Load Assets =======================
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_pca():
    return joblib.load("pca.pkl")

@st.cache_resource
def load_keras_model(input_shape):
    model = build_model(input_shape)
    model.load_weights("best_model.h5")  # Make sure this is saved correctly
    return model

scaler = load_scaler()
pca = load_pca()
model = load_keras_model(pca.n_components_)

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

pca_labels = get_pca_labels(pca, feature_names)

# ======================= App UI =======================
st.markdown("<div class='main-title'>🧠 Parkinson's Disease Classifier</div>", unsafe_allow_html=True)
st.write("Enter PCA-reduced input features to predict the likelihood of Parkinson's symptoms.")

# ======================= Grouped PCA Feature Inputs =======================
# ======================= Grouped PCA Feature Inputs in 3 Columns =======================
pca_input_features = []
st.subheader("Enter values for PCA-reduced features:")

with st.expander("🔧 PCA Feature Inputs", expanded=True):
    cols = st.columns(3)
    for i, label in enumerate(pca_labels):
        col = cols[i % 3]
        with col:
            with st.expander(label, expanded=False):
                value = st.number_input(
                    label=f"Value for {label.split('(')[0].strip()}",
                    value=0.0,
                    step=0.01,
                    key=f"pca_{i}"
                )
                pca_input_features.append(value)



# ======================= Prediction =======================
if st.button("🔍 Predict"):
    input_pca = np.array(pca_input_features).reshape(1, -1)
    prediction = model.predict(input_pca)[0][0]

    label = "Symptoms of Parkinson's" if prediction > 0.5 else "a Healthy Body"
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    color = "#ff4b4b" if prediction > 0.5 else "#28a745"

    st.markdown(f"""
        <div class="prediction-box">
            <b>Prediction:</b> <span style="color:{color}">{label}</span><br>
            <b>Confidence:</b> {confidence:.2f}%
        </div>
    """, unsafe_allow_html=True)
