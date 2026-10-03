import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import os
import joblib


def save_model(pipeline, file_path):
    folder = os.path.dirname(file_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    joblib.dump(pipeline, file_path)

    print(f"Model saved successfully to: {file_path}")


def prepare_ml_data(data):
    ml_data = data.copy()

    ml_data = ml_data[
        ['type', 'release_year', 'rating', 'country', 'listed_in']
    ]

    ml_data = ml_data.dropna()

    return ml_data

def split_features_target(data):
    X = data.drop('type', axis=1)
    y = data['type']

    return X, y

def create_train_test_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def create_preprocessor():
    categorical_columns = [
        'rating',
        'country',
        'listed_in'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'categorical',
                OneHotEncoder(handle_unknown='ignore'),
                categorical_columns
            )
        ],
        remainder='passthrough'
    )

    return preprocessor

def create_model_pipeline():
    preprocessor = create_preprocessor()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessing', preprocessor),
            ('model', model)
        ]
    )

    return pipeline

def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)

    return pipeline

def make_predictions(pipeline, X_test):
    predictions = pipeline.predict(X_test)

    return predictions

def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label='Movie'
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label='Movie'
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label='Movie'
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def create_confusion_matrix(y_test, predictions):
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=['Movie', 'TV Show']
    )

    return matrix

def generate_classification_report(y_test, predictions):
    report = classification_report(
        y_test,
        predictions
    )

    return report