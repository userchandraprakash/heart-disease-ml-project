"""
Data Loader Module
==================
Loads the Heart Disease dataset. Uses sklearn's built-in dataset or generates
a synthetic version so the project works out-of-the-box without downloading files.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


# Feature names matching the UCI Heart Disease dataset
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

TARGET_NAME = "target"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_heart_data(filepath: "str | None" = None) -> pd.DataFrame:
    """
    Load the heart disease dataset.

    If a CSV file path is provided, loads from that file.
    Otherwise, tries to load from the default data directory.
    If no file is found, generates a realistic synthetic dataset.

    Parameters
    ----------
    filepath : str or None
        Path to a CSV file. If None, uses default location or generates data.

    Returns
    -------
    pd.DataFrame
        DataFrame with heart disease features and target column.
    """
    # Try loading from provided path
    if filepath and os.path.exists(filepath):
        print(f"[INFO] Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        return df

    # Try loading from default location
    default_path = os.path.join(DATA_DIR, "heart.csv")
    if os.path.exists(default_path):
        print(f"[INFO] Loading data from: {default_path}")
        df = pd.read_csv(default_path)
        return df

    # Generate synthetic data that mimics the real dataset
    print("[INFO] No dataset file found. Generating realistic synthetic data...")
    df = _generate_synthetic_heart_data()

    # Save for future runs
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(default_path, index=False)
    print(f"[INFO] Synthetic data saved to: {default_path}")

    return df


def _generate_synthetic_heart_data(n_samples: int = 303, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mimics the UCI Heart Disease dataset.

    The generated data has realistic distributions for each feature,
    matching the statistical properties of the original Cleveland dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic heart disease DataFrame.
    """
    rng = np.random.RandomState(random_state)

    # Generate target (roughly balanced, slight majority with disease)
    target = rng.binomial(1, 0.54, n_samples)

    data = {}

    # age: 29-77, mean ~54
    data["age"] = rng.normal(54, 9, n_samples).clip(29, 77).astype(int)

    # sex: 0 or 1, ~68% male
    data["sex"] = rng.binomial(1, 0.68, n_samples)

    # cp (chest pain type): 0-3, type 0 most common in disease group
    probs_cp = np.where(
        target == 1,
        rng.choice([0, 1, 2, 3], n_samples, p=[0.15, 0.15, 0.30, 0.40]),
        rng.choice([0, 1, 2, 3], n_samples, p=[0.45, 0.20, 0.20, 0.15]),
    )
    data["cp"] = probs_cp.astype(int)

    # trestbps (resting blood pressure): 94-200, mean ~131
    data["trestbps"] = rng.normal(131, 17, n_samples).clip(94, 200).astype(int)

    # chol (serum cholesterol): 126-564, mean ~246
    data["chol"] = rng.normal(246, 52, n_samples).clip(126, 564).astype(int)

    # fbs (fasting blood sugar > 120): 0 or 1, ~15% have high fbs
    data["fbs"] = rng.binomial(1, 0.15, n_samples)

    # restecg: 0-2
    data["restecg"] = rng.choice([0, 1, 2], n_samples, p=[0.48, 0.48, 0.04])

    # thalach (max heart rate): higher in no-disease group
    base_thalach = np.where(target == 1, 160, 140)
    data["thalach"] = (base_thalach + rng.normal(0, 20, n_samples)).clip(71, 202).astype(int)

    # exang (exercise induced angina): more common in disease group
    exang_prob = np.where(target == 1, 0.15, 0.55)
    data["exang"] = rng.binomial(1, exang_prob)

    # oldpeak: higher in disease group
    base_oldpeak = np.where(target == 1, 0.6, 1.6)
    data["oldpeak"] = (base_oldpeak + rng.exponential(0.5, n_samples)).clip(0, 6.2).round(1)

    # slope: 0-2
    data["slope"] = rng.choice([0, 1, 2], n_samples, p=[0.20, 0.47, 0.33])

    # ca (number of major vessels): 0-3
    ca_probs = np.where(
        target == 1,
        rng.choice([0, 1, 2, 3], n_samples, p=[0.60, 0.22, 0.12, 0.06]),
        rng.choice([0, 1, 2, 3], n_samples, p=[0.25, 0.30, 0.25, 0.20]),
    )
    data["ca"] = ca_probs.astype(int)

    # thal: 1-3
    data["thal"] = rng.choice([1, 2, 3], n_samples, p=[0.06, 0.55, 0.39])

    data["target"] = target

    df = pd.DataFrame(data)
    return df


def get_feature_info() -> dict:
    """
    Return a dictionary with metadata about each feature.

    Returns
    -------
    dict
        Feature names mapped to their descriptions and types.
    """
    return {
        "age": {"description": "Age in years", "type": "numerical"},
        "sex": {"description": "Sex (1=male, 0=female)", "type": "categorical"},
        "cp": {"description": "Chest pain type (0-3)", "type": "categorical"},
        "trestbps": {"description": "Resting blood pressure (mm Hg)", "type": "numerical"},
        "chol": {"description": "Serum cholesterol (mg/dl)", "type": "numerical"},
        "fbs": {"description": "Fasting blood sugar > 120 mg/dl", "type": "categorical"},
        "restecg": {"description": "Resting ECG results (0-2)", "type": "categorical"},
        "thalach": {"description": "Max heart rate achieved", "type": "numerical"},
        "exang": {"description": "Exercise-induced angina", "type": "categorical"},
        "oldpeak": {"description": "ST depression by exercise", "type": "numerical"},
        "slope": {"description": "Slope of peak exercise ST", "type": "categorical"},
        "ca": {"description": "Major vessels colored by fluoroscopy (0-3)", "type": "categorical"},
        "thal": {"description": "Thalassemia type", "type": "categorical"},
    }


if __name__ == "__main__":
    df = load_heart_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
