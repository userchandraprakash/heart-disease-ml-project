"""
Preprocessing Module
====================
Handles data cleaning, feature engineering, scaling, and train/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Perform data quality checks and return a summary report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    dict
        Data quality report with missing values, duplicates, and basic stats.
    """
    report = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"  Shape              : {report['shape']}")
    print(f"  Total missing      : {report['total_missing']}")
    print(f"  Duplicate rows     : {report['duplicates']}")
    print(f"  Features           : {df.shape[1] - 1}")
    print("=" * 60)

    if report["total_missing"] > 0:
        print("\n  Missing values per column:")
        for col, count in report["missing_values"].items():
            if count > 0:
                print(f"    {col}: {count}")

    return report


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Strategy:
    - Numerical columns: fill with median
    - Categorical columns: fill with mode

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df = df.copy()

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [FIX] Filled '{col}' missing values with median: {median_val}")

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  [FIX] Filled '{col}' missing values with mode: {mode_val}")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)

    if n_removed > 0:
        print(f"  [FIX] Removed {n_removed} duplicate rows.")
    else:
        print("  [OK]  No duplicate rows found.")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones to improve model performance.

    New features:
    - age_group: Categorized age into groups
    - heart_rate_reserve: Estimated heart rate reserve
    - chol_level: Categorized cholesterol level

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features.
    """
    df = df.copy()

    # Age groups: young (<40), middle (40-55), senior (55-65), elderly (>65)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 55, 65, 100],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # Heart rate reserve (estimated max HR - resting HR approximation)
    # Max HR estimate = 220 - age
    df["hr_reserve"] = (220 - df["age"]) - df["thalach"]

    # Cholesterol level categories
    df["chol_level"] = pd.cut(
        df["chol"],
        bins=[0, 200, 240, 600],
        labels=[0, 1, 2],  # normal, borderline, high
    ).astype(int)

    # Blood pressure category
    df["bp_category"] = pd.cut(
        df["trestbps"],
        bins=[0, 120, 140, 300],
        labels=[0, 1, 2],  # normal, elevated, high
    ).astype(int)

    print(f"  [OK]  Engineered 4 new features: age_group, hr_reserve, chol_level, bp_category")
    return df


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    add_features: bool = True,
) -> tuple:
    """
    Full preprocessing pipeline: clean, engineer, split, and scale the data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with features and 'target' column.
    test_size : float
        Fraction of data to use for testing (default 0.2).
    random_state : int
        Random seed for reproducibility.
    add_features : bool
        Whether to add engineered features.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Data quality check
    print("\n📋 Step 1: Data Quality Check")
    check_data_quality(df)

    # Step 2: Handle missing values
    print("\n🔧 Step 2: Handle Missing Values")
    df = handle_missing_values(df)

    # Step 3: Remove duplicates
    print("\n🔧 Step 3: Remove Duplicates")
    df = remove_duplicates(df)

    # Step 4: Feature engineering
    if add_features:
        print("\n⚙️  Step 4: Feature Engineering")
        df = engineer_features(df)

    # Step 5: Split features and target
    print("\n✂️  Step 5: Train/Test Split")
    X = df.drop("target", axis=1)
    y = df["target"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Training set : {X_train.shape[0]} samples")
    print(f"  Testing set  : {X_test.shape[0]} samples")

    # Step 6: Feature scaling
    print("\n📏 Step 6: Feature Scaling (StandardScaler)")
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    print("  [OK]  Features scaled to zero mean and unit variance.")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    from data_loader import load_heart_data

    df = load_heart_data()
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)
    print(f"\nFeatures used ({len(features)}): {features}")
