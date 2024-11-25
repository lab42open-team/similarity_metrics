#!/usr/bin/python3.5
import joblib

model_path = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/logistic_regression/model_v5.0_LSU.pkl"
scaler_path = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/logistic_regression/scaler_v5.0_LSU.pkl"

model = joblib.load(model_path)
print(model)

scaler = joblib.load(scaler_path)
print(scaler)