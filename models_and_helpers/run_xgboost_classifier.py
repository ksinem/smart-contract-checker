import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
import json
import xgboost as xgb
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, ADASYN
from imblearn.combine import SMOTEENN
from utils import (
    save_embedding_vectors,
    generate_pandas_report)




with open("../config.json", "r") as c:
    config = json.load(c)


def prepare_data_for_xgboost():
    """
        Loads smart contract data, retrieves pre-computed code embeddings,
        and prepares the feature matrix (X) and target vector (y) for modeling.

        The function reads a CSV file containing smart contract information and
        loads corresponding Word2Vec-style embeddings from individual .npy files
        based on the file path specified in the config.

        Returns
        -------
        X : numpy.ndarray
            The feature matrix, where each row is a 300-dimensional vector
            representing an encoded smart contract. Shape is (N, 300).
        y : numpy.ndarray
            The target vector containing integer-encoded vulnerability labels.
            Shape is (N,).
        """

    sc_8_labels = pd.read_csv(str(config["data_path"]+config["contract8labels"]), index_col=0)
    #generate_pandas_report(sc_8_labels, "Smart contracts with 8 vulnerabilites")
    print(f"Total length of data: {len(sc_8_labels)}. \n")

    # save embedding vectors with function from utils.py before loading
    idx_to_encoding = {}
    for idx, code in enumerate(sc_8_labels["code"]):
        idx_to_encoding[idx] = np.load(f"{str(config['data_path'])}wv_encodings/code_{idx}_encoded.npy")

    X = sc_8_labels['code_encoded'] = pd.Series(idx_to_encoding)
    X = np.stack(X.values)
    # X is 4285x300 vector
    # it comprises one 300d-vec per code
    y = sc_8_labels['label_encoded'].values.astype(np.int32)


    print(f"Data preparation successful.\n"
          f"X-Shape: {X.shape}, y-Shape: {y.shape}")

    return X, y


def train_and_evaluate_xgboost():
    """
    Prepares the data, splits it, applies Random Oversampling to the training set,
    trains an XGBoost model, and evaluates its performance on the test set.

    The function uses parameters defined in the global 'config' and prints
    the data balance status, training progress, final accuracy, and a
    full classification report.

    Note: Random Oversampling is used to address the class imbalance issue
    by duplicating minority class samples.
    """
    X, y = prepare_data_for_xgboost()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Starting Random Oversampling to balance out the data...")
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
    print(f"Oversampling done. New number of instances: {X_resampled.shape[0]}\n"
          f"with {pd.Series(y_resampled).value_counts()[0]} instances per label. "
          f"X-Shape: {X_resampled.shape[0]}, y-Shape: {y_resampled.shape}")

    print("\n--- 1. Training  ---")
    model_params = config["xgb_params"]
    model = XGBClassifier(**model_params)
    model.fit(X_resampled, y_resampled)
    print("Training done.")

    y_pred = model.predict(X_test)
    print("\n--- 2. Evaluation ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    report = classification_report(
        y_test,
        y_pred)
    print("\n--- Classification report:")
    print(report)
    print("-----------------------------------")


if __name__ == "__main__":
    train_and_evaluate_xgboost()

