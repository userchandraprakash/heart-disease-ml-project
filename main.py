#!/usr/bin/env python3
"""
Heart Disease Prediction — Machine Learning Pipeline
=====================================================

Main entry point for the project. Runs the full ML pipeline:
1. Load dataset
2. Exploratory Data Analysis
3. Preprocess data
4. Train & evaluate multiple models
5. Generate visualizations
6. Save best model

Usage:
    python main.py

Author: AI/ML Student Project
"""

import sys
import time
from src.data_loader import load_heart_data
from src.preprocessing import preprocess_data
from src.eda import run_eda
from src.models import train_and_evaluate_all
from src.visualizations import generate_all_result_plots


def print_banner():
    """Print the project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     ❤️  Heart Disease Prediction using ML  ❤️             ║
    ║                                                          ║
    ║     Comparing 6 ML Models with Hyperparameter Tuning     ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Run the complete ML pipeline."""
    start_time = time.time()
    print_banner()

    # ────────────────────────────────────────────
    # Step 1: Load Data
    # ────────────────────────────────────────────
    print("\n🔹 STEP 1: Loading Dataset")
    print("─" * 40)
    df = load_heart_data()
    print(f"  Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

    # ────────────────────────────────────────────
    # Step 2: Exploratory Data Analysis
    # ────────────────────────────────────────────
    print("\n🔹 STEP 2: Exploratory Data Analysis")
    print("─" * 40)
    run_eda(df)

    # ────────────────────────────────────────────
    # Step 3: Preprocess Data
    # ────────────────────────────────────────────
    print("\n🔹 STEP 3: Data Preprocessing")
    print("─" * 40)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

    # ────────────────────────────────────────────
    # Step 4: Train & Evaluate Models
    # ────────────────────────────────────────────
    print("\n🔹 STEP 4: Model Training & Evaluation")
    print("─" * 40)
    results = train_and_evaluate_all(X_train, X_test, y_train, y_test, tune=True)

    # ────────────────────────────────────────────
    # Step 5: Generate Visualizations
    # ────────────────────────────────────────────
    print("\n🔹 STEP 5: Generating Visualizations")
    print("─" * 40)
    generate_all_result_plots(results, y_test, feature_names)

    # ────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time     : {elapsed:.1f} seconds")
    print(f"  Models trained : {len(results)}")
    print(f"  Best model     : {max(results, key=lambda k: results[k]['accuracy'])}")
    print(f"  Plots saved to : results/plots/")
    print(f"  Model saved to : models/best_model.joblib")
    print("=" * 60)

    # ────────────────────────────────────────────
    # Demo: Single Prediction
    # ────────────────────────────────────────────
    print("\n🔮 DEMO: Single Patient Prediction")
    print("─" * 40)

    from src.models import predict_single

    sample_patient = {
        "age": 55, "sex": 1, "cp": 2, "trestbps": 140, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2,
        "age_group": 2, "hr_reserve": 15, "chol_level": 2, "bp_category": 1,
    }

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    result = predict_single(best_model, scaler, sample_patient)

    print(f"  Patient age    : {sample_patient['age']}")
    print(f"  Prediction     : {result['diagnosis']}")
    print(f"  Confidence     : {result['confidence']:.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
