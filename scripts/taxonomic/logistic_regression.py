#!/usr/bin/python3.5

# script name: logistic_regression.py
# developed by: Nefeli Venetsianou
# description: 
    # Supervised machine learning to predict biome per sample.
# framework: CCMRI
# last update: 07/11/2024

import os 
import joblib
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def load_data(input_file, biome_file):
    try:
        # Load input file (cosine similarity metrics file)
        df = pd.read_csv(input_file, sep="\t")
        print(df)
        # Load biome information
        biome_df = pd.read_csv(biome_file, sep="\t")
        # Create dictionary to map sample_id with biome_info
        biome_dict = dict(zip(biome_df["sample_id"], biome_df["biome_info"]))
        # Get the unique sample names
        samples = pd.concat([df["Sample1"], df["Sample2"]]).unique()
        # Initialize the symmetric matrix
        distance_matrix = pd.DataFrame(np.ones((len(samples), len(samples))), index=samples, columns=samples)
        # Fill the matrix with distances from the pairwise data 
        for _, row in df.iterrows():
            sample1 = row["Sample1"]
            sample2 = row["Sample2"]
            distance = row["Distance"]
            distance_matrix.loc[sample1, sample2] = distance
            distance_matrix.loc[sample2, sample1] = distance  # Make it symmetric(each sample pair has distance entry)
        # Add biome info to the distance matrix
        distance_matrix["biome_info"] = distance_matrix.index.map(biome_dict).fillna("Unknown")
        # Count occurrences of each biome
        biome_counts = distance_matrix["biome_info"].value_counts()
        # Print the unique biomes and their counts
        logging.info("Unique biomes and their counts:")
        for biome, count in biome_counts.items():
            logging.info("{}: {}".format(biome, count))
        # Verify the matrix
        logging.info("Distance matrix loaded and filled successfully:\n{}.".format(distance_matrix.head()))
        return distance_matrix
    except Exception as e:
        logging.error("Error in loading and preprocessing data: {}".format(e))
        raise

def preprocess_data(distance_matrix):
    try:
        logging.info("Preprocessing data...")
        # Separate features (x) and labels (y)
        x = distance_matrix.drop(columns=["biome_info"]) # Drop biome info
        y = distance_matrix["biome_info"]
        # Factorize labels and store the mapping for later use
        unique_biomes, y_encoded = np.unique(y, return_inverse=True) 
        label_mapping = {index: label for index, label in enumerate(unique_biomes)}
        logging.info("Class label mapping: {}".format(label_mapping))
        return x, y_encoded, label_mapping
    except Exception as e:
        logging.error("Error in preprocessing data: {}".format(e))
        raise

def split_data(x, y, test_size = 0.2, random_state = 42):
    try:
        # x matrix contains features (distances) - y contains biome classes (prediction) 
        logging.info("Splitting data to train and test sets...")
        # 80% training -20% testing, random_state controls the shuffling applied to the data before applying the split.
        # random_state = int - allows to get the same training/testing split each time you run the code - 
        # randomly reordering the samples in the dataset before dividing them into training and testing sets - any integer would work.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error("Error in splittind data: {}".format(e))
        raise

def standarize_data(x_train, x_test):
    try:
        logging.info("Standarizing features...")
        # Normalize feature values for consistent scale accross features
        # Standardize features by removing the mean and scaling to unit variance. The standard score of a sample `x` is calculated as: z = (x - u) / s
        scaler = StandardScaler() 
        x_train = scaler.fit_transform(x_train) # Fit to data, then transform it.
        x_test = scaler.transform(x_test) # Fit to data.
        return x_train, x_test, scaler
    except Exception as e:
        logging.error("Error in standarizing features: {}".format(e))
        raise

def train_model(x_train, y_train):
    try:
        logging.info("Training data...")
        # model learns relationships from x_train to predict y_train - learn the association of distances / biome info (80%)
        # model predicts labels (y_test) given distances (x_test)
        model = LogisticRegression(multi_class="multinomial") # even without multi_class="multinomial", it chooses automatically by default the best strategy based on No of labels 
        model.fit(x_train, y_train)
        return model
    except Exception as e:
        logging.error("Error in training the modeil: {}".format(e))
        raise

def evaluate_model(model, x_test, y_test, label_mapping, output_file):
    try:
        logging.info("Evaluating model...")
        # Predict and map predictions back to original labels
        y_pred_encoded = model.predict(x_test)
        y_pred = [label_mapping[label] for label in y_pred_encoded]
        y_test_decoded = [label_mapping[label] for label in y_test]
        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_test_decoded, y_pred) # In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must *exactly* match the corresponding set of labels in y_true.
        report = classification_report(y_test_decoded, y_pred, zero_division=1) # Build a text report showing the main classification metrics
        logging.info("Model Accuracy: {:2f}%".format(accuracy * 100))
        logging.info("Classification Report:\n" + report)
        # Save accuracy score and report to output txt file
        with open(output_file, "w") as f:
            f.write("Model Accuracy: {:2f}%\n\n".format(accuracy * 100))
            f.write("Classification Report: \n {}")
            f.write(report)
        return accuracy, report
    except Exception as e:
        logging.error("Error in evaluating data: {}".format(e))
        raise

def save_model_and_scaler(model, scaler, model_path, scaler_path):
    try:
        logging.info("Saving model and scaler...")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
    except Exception as e:
        logging.error("Error in saving model and scaler: {}".format(e))
        raise

from sklearn.model_selection import cross_val_score

def cross_validate_model(x, y, label_mapping, output_file, cv=3):
    try:
        logging.info("Performing cross-validation...")
        model = LogisticRegression(multi_class="multinomial")
        # Perform cross-validation
        scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy')
        average_accuracy = np.mean(scores)
        logging.info("Cross-Validation Accuracy Scores: {}".format(scores))
        logging.info("Average Accuracy: {:2f}%".format(average_accuracy * 100))
        # Write results to output file
        with open(output_file, "w") as f:
            f.write("Cross-Validation Accuracy Scores: {}\n".format(scores))
            f.write("Average Accuracy: {:2f}%\n".format(average_accuracy * 100))
        return average_accuracy
    except Exception as e:
        logging.error("Error in cross-validation: {}".format(e))
        raise

def main():
    data_dir = "/ccmri/similarity_metrics/data/taxonomic/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/"
    biome_dir = "/ccmri/similarity_metrics/data/taxonomic/taxonomic/raw_data/studies_samples/biome_info/3b"
    data_file = os.path.join(data_dir, "c_distances_filtered_v5.0_LSU_ge_filtered.tsv")
    biome_file = os.path.join(biome_dir, "study_sample_3biome_v5.0_LSU.tsv")
    model_path = "/ccmri/similarity_metrics/data/taxonomic/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/logistic_regression/test_model_v5.0_LSU.pkl"
    scaler_path = "/ccmri/similarity_metrics/data/taxonomic/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/logistic_regression/test_scaler_v5.0_LSU.pkl"
    evaluation_output_file = "/ccmri/similarity_metrics/data/taxonomic/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/logistic_regression/evaluation_output.txt"
    # Load data
    distance_matrix = load_data(data_file, biome_file)
    # Preprocess data to separate features x, y
    x, y_encoded, label_mapping = preprocess_data(distance_matrix)
    # Print class mappings
    logging.info("Class mappings: {}".format(label_mapping))
    # Split data to training and testing sets
    x_train, x_test, y_train, y_test = split_data(x,y_encoded)
    # Standarize data - normalize features 
    x_train, x_test, scaler = standarize_data(x_train, x_test)
    # Train model 
    model = train_model(x_train, y_train)
    # Evaluate model 
    accuracy_score, report = evaluate_model(model, x_test, y_test, label_mapping, evaluation_output_file)
    # Save model and scaler 
    save_model_and_scaler(model, scaler, model_path, scaler_path)
    logging.info("Script completed successfuly.")
    # Perform cross-validation and save results
    cross_validate_model(x, y_encoded, label_mapping, evaluation_output_file, cv=3)


if __name__ == "__main__":
    main()
    