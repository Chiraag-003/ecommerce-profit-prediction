import pickle
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


NUMERICAL_COLUMNS = ["quantity", "sales"]

MODEL_FILE = "profit_prediction_model.pkl"
SCALER_FILE = "scaler.pkl"
FEATURES_FILE = "features.pkl"


@st.cache_resource
def load_artifacts():
    base_path = Path(__file__).resolve().parent

    model = joblib.load(base_path / MODEL_FILE)
    scaler = joblib.load(base_path / SCALER_FILE)

    with open(base_path / FEATURES_FILE, "rb") as f:
        features = pickle.load(f)

    categories = sorted([f.replace("category_", "") for f in features if f.startswith("category_")])
    regions = sorted([f.replace("region_", "") for f in features if f.startswith("region_")])

    return model, scaler, features, categories, regions


def build_input(scaler, features, category, region, quantity, sales):
    input_df = pd.DataFrame([{
        "category": category,
        "region": region,
        "quantity": quantity,
        "sales": sales
    }])

    num_scaled = scaler.transform(input_df[NUMERICAL_COLUMNS])
    num_df = pd.DataFrame(num_scaled, columns=NUMERICAL_COLUMNS)

    cat_df = pd.get_dummies(input_df[["category", "region"]])

    X = pd.concat([num_df, cat_df], axis=1)

    return X.reindex(columns=features, fill_value=0)


def predict(model, scaler, features, category, region, quantity, sales):
    X = build_input(scaler, features, category, region, quantity, sales)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    return pred, prob


st.title("📊 E-commerce Profit Prediction System")

model, scaler, features, categories, regions = load_artifacts()

category = st.selectbox("Category", categories)
region = st.selectbox("Region", regions)

quantity = st.number_input("Quantity", min_value=1.0, value=10.0)
sales = st.number_input("Sales", min_value=0.0, value=100.0)

if st.button("Predict"):

    prediction, probability = predict(
        model,
        scaler,
        features,
        category,
        region,
        quantity,
        sales
    )

    if prediction == 1:
        st.success(f"High Profit ✅ ({probability*100:.2f}%)")
    else:
        st.error(f"Low Profit ❌ ({probability*100:.2f}%)")