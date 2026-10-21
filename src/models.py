"""
Models Module
=============
Trains, evaluates, and compares multiple ML classifiers for heart disease prediction.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def get_models() -> dict:
    """
    Return a dictionary of ML models to train and compare.

    Returns
    -------
    dict
        Model name -> sklearn estimator.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        ),
    }


def train_model(model, X_train, y_train, model_name: str = "Model"):
    """
    Train a single model and print training info.

    Parameters
    ----------
    model : sklearn estimator
        The model to train.
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training labels.
    model_name : str
        Name of the model for display.

    Returns
    -------
    sklearn estimator
        The trained model.
    """
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    print(f"  ✅ {model_name:28s} | Train Accuracy: {train_acc:.4f}")
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a trained model on test data.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    X_test : pd.DataFrame or np.ndarray
        Test features.
    y_test : pd.Series or np.ndarray
        True labels.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics


def cross_validate_model(model, X, y, cv: int = 5) -> dict:
    """
    Perform cross-validation on a model.

    Parameters
    ----------
    model : sklearn estimator
        Model to validate.
    X : pd.DataFrame or np.ndarray
        Full feature set.
    y : pd.Series or np.ndarray
        Full target.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    dict
        Cross-validation scores summary.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return {
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "cv_scores": scores,
    }


def hyperparameter_tuning(X_train, y_train) -> dict:
    """
    Perform hyperparameter tuning on selected models using GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    dict
        Best models and their parameters.
    """
    print("\n🔍 Hyperparameter Tuning with GridSearchCV...")

    param_grids = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
            },
        },
    }

    best_models = {}

    for name, config in param_grids.items():
        print(f"\n  Tuning {name}...")
        grid_search = GridSearchCV(
            config["model"],
            config["params"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        best_models[name] = {
            "model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
        }
        print(f"  Best params : {grid_search.best_params_}")
        print(f"  Best CV acc : {grid_search.best_score_:.4f}")

    return best_models


def train_and_evaluate_all(X_train, X_test, y_train, y_test, tune: bool = True) -> dict:
    """
    Train, evaluate, and compare all models.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Training and test features.
    y_train, y_test : pd.Series
        Training and test labels.
    tune : bool
        Whether to run hyperparameter tuning.

    Returns
    -------
    dict
        Results for each model with metrics and the trained estimator.
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    models = get_models()
    results = {}

    # Train all models
    print("\n📚 Training Models...")
    trained_models = {}
    for name, model in models.items():
        trained_models[name] = train_model(model, X_train, y_train, name)

    # Evaluate all models
    print("\n📊 Evaluating Models on Test Set...\n")
    print(f"  {'Model':28s} | {'Accuracy':>8s} | {'Precision':>9s} | {'Recall':>6s} | {'F1':>6s} | {'ROC-AUC':>7s}")
    print("  " + "-" * 80)

    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test)
        cv_results = cross_validate_model(model, X_train, y_train)

        results[name] = {
            "model": model,
            **metrics,
            **cv_results,
        }

        roc_str = f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "  N/A "
        print(
            f"  {name:28s} | {metrics['accuracy']:>8.4f} | {metrics['precision']:>9.4f} | "
            f"{metrics['recall']:>6.4f} | {metrics['f1_score']:>6.4f} | {roc_str:>7s}"
        )

    # Hyperparameter tuning
    if tune:
        tuned = hyperparameter_tuning(X_train, y_train)
        for name, info in tuned.items():
            tuned_metrics = evaluate_model(info["model"], X_test, y_test)
            tuned_name = f"{name} (Tuned)"
            results[tuned_name] = {
                "model": info["model"],
                **tuned_metrics,
                "best_params": info["best_params"],
            }
            roc_str = f"{tuned_metrics['roc_auc']:.4f}" if tuned_metrics["roc_auc"] else "  N/A "
            print(
                f"  {tuned_name:28s} | {tuned_metrics['accuracy']:>8.4f} | {tuned_metrics['precision']:>9.4f} | "
                f"{tuned_metrics['recall']:>6.4f} | {tuned_metrics['f1_score']:>6.4f} | {roc_str:>7s}"
            )

    # Find and save best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    best_acc = results[best_name]["accuracy"]

    print(f"\n🏆 Best Model: {best_name} (Accuracy: {best_acc:.4f})")

    # Save best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"  [SAVED] Best model saved to: {model_path}")

    # Detailed classification report for best model
    print(f"\n📋 Classification Report — {best_name}:")
    print(classification_report(y_test, results[best_name]["y_pred"], target_names=["No Disease", "Disease"]))

    return results


def load_saved_model(filepath: str = None):
    """
    Load a previously saved model.

    Parameters
    ----------
    filepath : str or None
        Path to the saved model. Defaults to 'models/best_model.joblib'.

    Returns
    -------
    sklearn estimator
        The loaded model.
    """
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, "best_model.joblib")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved model found at: {filepath}")

    model = joblib.load(filepath)
    print(f"[INFO] Model loaded from: {filepath}")
    return model


def predict_single(model, scaler, feature_values: dict) -> dict:
    """
    Make a prediction for a single patient.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    scaler : StandardScaler
        Fitted scaler.
    feature_values : dict
        Dictionary of feature names to values.

    Returns
    -------
    dict
        Prediction result with probability.
    """
    df = pd.DataFrame([feature_values])
    X_scaled = scaler.transform(df)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None

    result = {
        "prediction": int(prediction),
        "diagnosis": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
        "confidence": float(max(probability)) if probability is not None else None,
    }

    return result
