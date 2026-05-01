import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


CATEGORICAL_COLUMNS = ["category", "region"]
NUMERICAL_COLUMNS = ["quantity", "sales"]
TARGET_COLUMN = "high_profit"


def clean_column_name(col):
    col = col.strip().lower()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^a-z0-9_]", "", col)
    return col


def load_data(path):
    df = pd.read_csv(path)
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def preprocess_data(df):
    df = df.dropna().copy()

    for col in NUMERICAL_COLUMNS + ["profit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    df[TARGET_COLUMN] = (df["profit"] > df["profit"].median()).astype(int)

    return df


def build_features(df, scaler=None):
    num_df = df[NUMERICAL_COLUMNS]

    if scaler is None:
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(num_df)
    else:
        num_scaled = scaler.transform(num_df)

    num_scaled = pd.DataFrame(num_scaled, columns=NUMERICAL_COLUMNS, index=df.index)

    cat_df = pd.get_dummies(df[CATEGORICAL_COLUMNS])

    X = pd.concat([num_scaled, cat_df], axis=1)

    return X, scaler


def train_model(df):
    X, scaler = build_features(df)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, preds))

    return model, scaler, list(X.columns)   # ✅ IMPORTANT FIX


if __name__ == "__main__":

    df = load_data("compressed_data (2).csv")

    df = preprocess_data(df)

    model, scaler, features = train_model(df)

    joblib.dump(model, "profit_prediction_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(features, "features.pkl")   # ✅ saves list, not Index

    print("\nModel saved successfully.")