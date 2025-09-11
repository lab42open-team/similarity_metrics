import joblib

model = joblib.load("/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/logistic_regression/test_model_v5.0.pkl")
scaler = joblib.load("/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/logistic_regression/test_scaler_v5.0.pkl")

print(f"Model: {model}")
print(f"Scaler: {scaler}")
