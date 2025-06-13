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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, label_binarize

logging.basicConfig(level=logging.INFO)

def load_data(input_file, biome_file):
    try:
        df = pd.read_csv(input_file, sep="\t")
        print(df)
        biome_df = pd.read_csv(biome_file, sep="\t")
        biome_dict = dict(zip(biome_df["sample_id"], biome_df["biome_info"]))
        samples = pd.concat([df["Sample1"], df["Sample2"]]).unique()
        distance_matrix = pd.DataFrame(np.ones((len(samples), len(samples))), index=samples, columns=samples)
        for _, row in df.iterrows():
            sample1 = row["Sample1"]
            sample2 = row["Sample2"]
            distance = row["Distance"]
            distance_matrix.loc[sample1, sample2] = distance
            distance_matrix.loc[sample2, sample1] = distance
        distance_matrix["biome_info"] = distance_matrix.index.map(biome_dict).fillna("Unknown")
        biome_counts = distance_matrix["biome_info"].value_counts()
        logging.info("Unique biomes and their counts:")
        for biome, count in biome_counts.items():
            logging.info("{}: {}".format(biome, count))
        logging.info("Distance matrix loaded and filled successfully:\n{}.".format(distance_matrix.head()))
        return distance_matrix
    except Exception as e:
        logging.error("Error in loading and preprocessing data: {}".format(e))
        raise

def preprocess_data(distance_matrix):
    try:
        logging.info("Preprocessing data...")
        x = distance_matrix.drop(columns=["biome_info"])
        y = distance_matrix["biome_info"]
        unique_biomes, y_encoded = np.unique(y, return_inverse=True) 
        label_mapping = {index: label for index, label in enumerate(unique_biomes)}
        logging.info("Class label mapping: {}".format(label_mapping))
        return x, y_encoded, label_mapping
    except Exception as e:
        logging.error("Error in preprocessing data: {}".format(e))
        raise

def split_data(x, y, test_size = 0.2, random_state = 42):
    try:
        logging.info("Splitting data to train and test sets...")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error("Error in splitting data: {}".format(e))
        raise

def standarize_data(x_train, x_test):
    try:
        logging.info("Standardizing features...")
        scaler = StandardScaler() 
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        return x_train, x_test, scaler
    except Exception as e:
        logging.error("Error in standardizing features: {}".format(e))
        raise

def train_model(x_train, y_train):
    try:
        logging.info("Training model...")
        model = LogisticRegression(multi_class="multinomial", max_iter=1000)
        model.fit(x_train, y_train)
        return model
    except Exception as e:
        logging.error("Error in training the model: {}".format(e))
        raise

def evaluate_model(model, x_test, y_test, label_mapping, output_file):
    try:
        logging.info("Evaluating model...")
        y_pred_encoded = model.predict(x_test)
        y_pred = [label_mapping[label] for label in y_pred_encoded]
        y_test_decoded = [label_mapping[label] for label in y_test]
        accuracy = accuracy_score(y_test_decoded, y_pred)
        report = classification_report(y_test_decoded, y_pred, zero_division=1)
        logging.info("Model Accuracy: {:2f}%".format(accuracy * 100))
        logging.info("Classification Report:\n" + report)
        with open(output_file, "w") as f:
            f.write("Model Accuracy: {:2f}%\n\n".format(accuracy * 100))
            f.write("Classification Report:\n")
            f.write(report)
        return accuracy, report
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise

def plot_precision_recall(model, x_test, y_test, label_mapping, output_path):
    try:
        logging.info("Plotting Precision-Recall curve...")
        y_score = model.predict_proba(x_test)
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        precision = dict()
        recall = dict()
        avg_precision = dict()

        plt.figure(figsize=(10, 8))
        for i in range(len(classes)):
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            avg_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
            plt.plot(recall[i], precision[i], lw=2,
                     label='Class {} (AP={:0.2f})'.format(label_mapping[i], avg_precision[i]))
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve per Class')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        logging.info("Precision-Recall curve saved at: {}".format(output_path))
    except Exception as e:
        logging.error("Error in plotting Precision-Recall curve: {}".format(e))
        raise

def save_model_and_scaler(model, scaler, model_path, scaler_path):
    try:
        logging.info("Saving model and scaler...")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
    except Exception as e:
        logging.error("Error in saving model and scaler: {}".format(e))
        raise

def cross_validate_model(x, y, label_mapping, output_file, cv=3):
    try:
        logging.info("Performing cross-validation...")
        model = LogisticRegression(multi_class="multinomial", max_iter=1000)
        scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy')
        avg_acc = np.mean(scores)
        logging.info("Cross-Validation Accuracy Scores: {}".format(scores))
        logging.info("Average Accuracy: {:2f}%".format(avg_acc * 100))
        with open(output_file, "a") as f:
            f.write("\nCross-Validation Accuracy Scores: {}\n".format(scores))
            f.write("Average Accuracy: {:2f}%\n".format(avg_acc * 100))
        return avg_acc
    except Exception as e:
        logging.error("Error in cross-validation: {}".format(e))
        raise

def main():
    data_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/"
    biome_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/studies_samples/biome_info/3b"
    data_file = os.path.join(data_dir, "c_distances_batches_filtered_v4.1_LSU_ge_filtered.tsv")
    biome_file = os.path.join(biome_dir, "study_sample_3biome_v4.1_LSU.tsv")
    model_path = os.path.join(data_dir, "logistic_regression/test_model_v4.1_LSU.pkl")
    scaler_path = os.path.join(data_dir, "logistic_regression/test_scaler_v4.1_LSU.pkl")
    evaluation_output_file = os.path.join(data_dir, "logistic_regression/evaluation_output_v4.1_LSU.txt")
    pr_curve_output_file = os.path.join(data_dir, "logistic_regression/precision_recall_curve_v4.1_LSU.png")

    distance_matrix = load_data(data_file, biome_file)
    x, y_encoded, label_mapping = preprocess_data(distance_matrix)
    x_train, x_test, y_train, y_test = split_data(x, y_encoded)
    x_train, x_test, scaler = standarize_data(x_train, x_test)
    model = train_model(x_train, y_train)
    accuracy, report = evaluate_model(model, x_test, y_test, label_mapping, evaluation_output_file)
    save_model_and_scaler(model, scaler, model_path, scaler_path)
    plot_precision_recall(model, x_test, y_test, label_mapping, pr_curve_output_file)
    cross_validate_model(x, y_encoded, label_mapping, evaluation_output_file, cv=3)
    logging.info("Script completed successfully.")

if __name__ == "__main__":
    main()
