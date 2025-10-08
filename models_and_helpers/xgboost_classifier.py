import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json
import os
#from utils import perform_random_oversampling
import xgboost as xgb
from xgboost import XGBClassifier
from utils import save_embedding_vectors




with open("../config.json", "r") as c:
    config = json.load(c)


def prepare_data_for_xgboost():

    sc_8_labels = pd.read_csv("../data_old/SC_Vuln_8label.csv", index_col=0)
    print(f"Gesamter Datensatz: {len(sc_8_labels)} Zeilen.")

    # save embedding vectors with function from utils.py before loading
    idx_to_encoding = {}
    for idx, code in enumerate(sc_8_labels["code"]):
        idx_to_encoding[idx] = np.load(f"../data_old/wv_encodings/code_{idx}_encoded.npy")

    X = sc_8_labels['code_encoded'] = pd.Series(idx_to_encoding)
    X = np.stack(X.values)
    # X is 4285x300 vector
    # it comprises one 300d-vec per code
    labels = sc_8_labels['label_encoded'].values.astype(np.int32)
    print(f"Daten erfolgreich vorbereitet. X-Shape: {X.shape}, y-Shape: {labels.shape}")

    return X, labels


def train_and_evaluate_xgboost():
    """
    Hauptfunktion zum Trainieren und Evaluieren des XGBoost-Modells.
    """
    X, y = prepare_data_for_xgboost()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n--- 1. Training des XGBoost-Modells (Gradient Boosting) ---")

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='multi:softmax',
        num_class=8,
        eval_metric='mlogloss',
        random_state=42
    )

    model.fit(X_train, y_train)
    print("Training abgeschlossen.")

    y_pred = model.predict(X_test)

    print("\n--- 2. Evaluierung des Modells ---")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Genauigkeit (Accuracy): {accuracy:.4f}")

    report = classification_report(
        y_test,
        y_pred)
    print("\nKlassifikationsbericht:")
    print(report)

    print("-----------------------------------")
    print("XGBoost-Modelltraining erfolgreich abgeschlossen.")


if __name__ == "__main__":
    train_and_evaluate_xgboost()