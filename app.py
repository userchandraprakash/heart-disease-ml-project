"""
Heart Disease Prediction — Interactive Web Dashboard
=====================================================
Run with: streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc, classification_report,
)

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="❤️ Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        color: white;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
        color: #f0f0f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .healthy {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
    }
    .disease {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────
@st.cache_data
def load_data():
    """Load or generate the heart disease dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "heart.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        # Generate synthetic data
        rng = np.random.RandomState(42)
        n = 303
        target = rng.binomial(1, 0.54, n)
        data = {
            "age": rng.normal(54, 9, n).clip(29, 77).astype(int),
            "sex": rng.binomial(1, 0.68, n),
            "cp": np.where(target == 1,
                rng.choice([0,1,2,3], n, p=[0.15,0.15,0.30,0.40]),
                rng.choice([0,1,2,3], n, p=[0.45,0.20,0.20,0.15])).astype(int),
            "trestbps": rng.normal(131, 17, n).clip(94, 200).astype(int),
            "chol": rng.normal(246, 52, n).clip(126, 564).astype(int),
            "fbs": rng.binomial(1, 0.15, n),
            "restecg": rng.choice([0,1,2], n, p=[0.48, 0.48, 0.04]),
            "thalach": (np.where(target==1, 160, 140) + rng.normal(0, 20, n)).clip(71, 202).astype(int),
            "exang": rng.binomial(1, np.where(target==1, 0.15, 0.55)),
            "oldpeak": (np.where(target==1, 0.6, 1.6) + rng.exponential(0.5, n)).clip(0, 6.2).round(1),
            "slope": rng.choice([0,1,2], n, p=[0.20, 0.47, 0.33]),
            "ca": np.where(target==1,
                rng.choice([0,1,2,3], n, p=[0.60,0.22,0.12,0.06]),
                rng.choice([0,1,2,3], n, p=[0.25,0.30,0.25,0.20])).astype(int),
            "thal": rng.choice([1,2,3], n, p=[0.06, 0.55, 0.39]),
            "target": target,
        }
        return pd.DataFrame(data)


@st.cache_data
def preprocess_and_train(df):
    """Preprocess data and train all models."""
    df2 = df.copy()
    # Feature engineering
    df2["age_group"] = pd.cut(df2["age"], bins=[0,40,55,65,100], labels=[0,1,2,3]).astype(int)
    df2["hr_reserve"] = (220 - df2["age"]) - df2["thalach"]
    df2["chol_level"] = pd.cut(df2["chol"], bins=[0,200,240,600], labels=[0,1,2]).astype(int)
    df2["bp_category"] = pd.cut(df2["trestbps"], bins=[0,120,140,300], labels=[0,1,2]).astype(int)

    X = df2.drop("target", axis=1)
    y = df2["target"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1] if hasattr(model, "predict_proba") else None

        cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="accuracy")

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else 0,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

    return results, X_test_sc, y_test, scaler, feature_names


# ─── Load Data & Train ───────────────────────────────────
df = load_data()
results, X_test, y_test, scaler, feature_names = preprocess_and_train(df)

# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/heart-with-pulse.png", width=80)
    st.title("Navigation")
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Samples", df.shape[0])
    st.metric("Features", df.shape[1] - 1)
    st.metric("Disease Cases", int(df["target"].sum()))
    st.metric("Healthy Cases", int((df["target"] == 0).sum()))

    st.markdown("---")
    st.markdown("### 🏆 Best Model")
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    st.success(f"**{best_name}**")
    st.metric("Accuracy", f"{results[best_name]['accuracy']:.2%}")

    st.markdown("---")
    st.markdown(
        "Made with ❤️ by **AI/ML Student**\n\n"
        "[GitHub](https://github.com) • "
        "[LinkedIn](https://linkedin.com)"
    )


# ─── Main Content ────────────────────────────────────────
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("❤️ Heart Disease Prediction using Machine Learning")
st.markdown("*Comparing 6 ML models with interactive visualizations and real-time predictions*")
st.markdown('</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🔍 EDA", "🤖 Model Results", "📈 Visualizations", "🔮 Predict"
])

# ═══════════════════════════════════════════════════════════
# TAB 1: Dashboard
# ═══════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Overview Dashboard")
    st.markdown("---")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>303</h3>
            <p>Total Patients</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>6</h3>
            <p>ML Models Trained</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>{results[best_name]['accuracy']:.1%}</h3>
            <p>Best Accuracy</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3>17</h3>
            <p>Features Used</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick comparison table
    st.subheader("📋 Quick Model Comparison")
    summary_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": f"{m['accuracy']:.2%}",
            "Precision": f"{m['precision']:.2%}",
            "Recall": f"{m['recall']:.2%}",
            "F1-Score": f"{m['f1_score']:.2%}",
            "ROC-AUC": f"{m['roc_auc']:.2%}",
            "CV Mean ± Std": f"{m['cv_mean']:.2%} ± {m['cv_std']:.2%}",
        }
        for name, m in results.items()
    ])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Dataset preview
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 2: EDA
# ═══════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Exploratory Data Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Target distribution
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#2ecc71", "#e74c3c"]
        counts = df["target"].value_counts()
        bars = ax.bar(["No Disease (0)", "Heart Disease (1)"], [counts[0], counts[1]],
                       color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, [counts[0], counts[1]]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val),
                    ha="center", fontweight="bold", fontsize=12)
        ax.set_ylabel("Count")
        ax.set_title("Heart Disease Distribution", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with col2:
        # Pie chart
        st.subheader("Target Proportion")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie([counts[0], counts[1]], labels=["No Disease", "Heart Disease"],
               colors=colors, autopct="%1.1f%%", startangle=90, explode=(0.05, 0.05), shadow=True,
               textprops={"fontsize": 12})
        ax.set_title("Disease vs No Disease", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    st.pyplot(fig)
    plt.close()

    # Feature distributions
    st.subheader("📊 Feature Distributions by Target")
    feature_choice = st.selectbox("Select Feature:", ["age", "trestbps", "chol", "thalach", "oldpeak"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for target_val, color, label in [(0, "#2ecc71", "No Disease"), (1, "#e74c3c", "Disease")]:
        subset = df[df["target"] == target_val][feature_choice]
        ax.hist(subset, bins=20, alpha=0.6, color=color, label=label, edgecolor="black", linewidth=0.3)
    ax.set_title(f"{feature_choice.upper()} Distribution by Target", fontweight="bold")
    ax.set_xlabel(feature_choice)
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # Categorical analysis
    st.subheader("📊 Categorical Features vs Target")
    cat_choice = st.selectbox("Select Categorical Feature:", ["sex", "cp", "fbs", "restecg", "exang", "slope"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ct = pd.crosstab(df[cat_choice], df["target"])
    ct.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="black", linewidth=0.5)
    ax.set_title(f"{cat_choice.upper()} vs Target", fontweight="bold")
    ax.legend(["No Disease", "Disease"])
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close()

    # Statistics
    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 3: Model Results
# ═══════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 Model Training Results")
    st.markdown("---")

    # Per-model details
    selected_model = st.selectbox("Select Model for Details:", list(results.keys()))
    metrics = results[selected_model]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1-Score", f"{metrics['f1_score']:.2%}")
    col5.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"],
                    cbar=False, annot_kws={"size": 16})
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix — {selected_model}", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with col2:
        # ROC curve for selected model
        st.subheader("ROC Curve")
        if metrics["y_proba"] is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC = {roc_auc_val:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
            ax.fill_between(fpr, tpr, alpha=0.1, color="#3498db")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {selected_model}", fontweight="bold")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

    # Cross-validation
    st.subheader("📊 Cross-Validation Score")
    st.info(f"**{selected_model}** — CV Accuracy: **{metrics['cv_mean']:.2%}** ± {metrics['cv_std']:.2%}")

    # Feature importance for tree models
    model_obj = metrics["model"]
    if hasattr(model_obj, "feature_importances_"):
        st.subheader("🌟 Feature Importance")
        importances = model_obj.feature_importances_
        feat_imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(feat_imp_df)))
        ax.barh(feat_imp_df["Feature"], feat_imp_df["Importance"], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {selected_model}", fontweight="bold")
        for i, (imp, feat) in enumerate(zip(feat_imp_df["Importance"], feat_imp_df["Feature"])):
            ax.text(imp + 0.002, i, f"{imp:.3f}", va="center", fontsize=9)
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════
# TAB 4: Visualizations
# ═══════════════════════════════════════════════════════════
with tab4:
    st.header("📈 Comparative Visualizations")
    st.markdown("---")

    # Model comparison bar chart
    st.subheader("📊 Model Performance Comparison")
    model_names = list(results.keys())
    metrics_list = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors_bar = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(model_names))
    width = 0.2
    for i, (metric, label, color) in enumerate(zip(metrics_list, metric_labels, colors_bar)):
        vals = [results[name][metric] for name in model_names]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, edgecolor="black", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("All Models — Metric Comparison", fontsize=16, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # All ROC curves
    st.subheader("📈 ROC Curves — All Models")
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    for (name, m), color in zip(results.items(), cm_colors):
        if m["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, m["y_proba"])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — All Models", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # All confusion matrices
    st.subheader("🔲 Confusion Matrices — All Models")
    n_models = len(results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.ravel()
    for i, (name, m) in enumerate(results.items()):
        sns.heatmap(m["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=axes_flat[i],
                    xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"], cbar=False)
        axes_flat[i].set_title(name, fontsize=10, fontweight="bold")
        axes_flat[i].set_xlabel("Predicted")
        axes_flat[i].set_ylabel("Actual")
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════
# TAB 5: Predict
# ═══════════════════════════════════════════════════════════
with tab5:
    st.header("🔮 Predict Heart Disease")
    st.markdown("Enter patient details below and click **Predict** to get a diagnosis.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("🎂 Age", 20, 80, 55)
        sex = st.selectbox("👤 Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        cp = st.selectbox("💔 Chest Pain Type", [
            ("Typical Angina (0)", 0),
            ("Atypical Angina (1)", 1),
            ("Non-anginal Pain (2)", 2),
            ("Asymptomatic (3)", 3),
        ], format_func=lambda x: x[0])
        trestbps = st.slider("🩺 Resting Blood Pressure (mm Hg)", 90, 200, 130)
        chol = st.slider("🧪 Serum Cholesterol (mg/dl)", 100, 600, 245)

    with col2:
        fbs = st.selectbox("🍬 Fasting Blood Sugar > 120", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        restecg = st.selectbox("📉 Resting ECG", [
            ("Normal (0)", 0),
            ("ST-T Wave Abnormality (1)", 1),
            ("Left Ventricular Hypertrophy (2)", 2),
        ], format_func=lambda x: x[0])
        thalach = st.slider("💓 Max Heart Rate Achieved", 70, 210, 150)
        exang = st.selectbox("🏃 Exercise Induced Angina", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])

    with col3:
        oldpeak = st.slider("📐 ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
        slope = st.selectbox("📈 Slope of Peak Exercise ST", [
            ("Upsloping (0)", 0),
            ("Flat (1)", 1),
            ("Downsloping (2)", 2),
        ], format_func=lambda x: x[0])
        ca = st.selectbox("🩻 Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("🧬 Thalassemia", [
            ("Normal (1)", 1),
            ("Fixed Defect (2)", 2),
            ("Reversible Defect (3)", 3),
        ], format_func=lambda x: x[0])

    # Choose model for prediction
    st.markdown("---")
    pred_model_name = st.selectbox("🤖 Select Model for Prediction:", list(results.keys()), index=list(results.keys()).index(best_name))

    if st.button("🔮 Predict Now", use_container_width=True, type="primary"):
        # Build feature vector
        patient = {
            "age": age, "sex": sex[1], "cp": cp[1], "trestbps": trestbps,
            "chol": chol, "fbs": fbs[1], "restecg": restecg[1], "thalach": thalach,
            "exang": exang[1], "oldpeak": oldpeak, "slope": slope[1], "ca": ca, "thal": thal[1],
            "age_group": 0 if age < 40 else (1 if age < 55 else (2 if age < 65 else 3)),
            "hr_reserve": (220 - age) - thalach,
            "chol_level": 0 if chol < 200 else (1 if chol < 240 else 2),
            "bp_category": 0 if trestbps < 120 else (1 if trestbps < 140 else 2),
        }

        patient_df = pd.DataFrame([patient])
        X_scaled = scaler.transform(patient_df)

        model_obj = results[pred_model_name]["model"]
        prediction = model_obj.predict(X_scaled)[0]
        proba = model_obj.predict_proba(X_scaled)[0] if hasattr(model_obj, "predict_proba") else None

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            confidence = proba[1] * 100 if proba is not None else 0
            st.markdown(f"""
            <div class="prediction-box disease">
                ⚠️ Heart Disease Detected<br>
                <span style="font-size: 1rem; font-weight: normal;">Confidence: {confidence:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.error("⚠️ The model predicts a **high risk** of heart disease. Please consult a doctor.")
        else:
            confidence = proba[0] * 100 if proba is not None else 0
            st.markdown(f"""
            <div class="prediction-box healthy">
                ✅ No Heart Disease Detected<br>
                <span style="font-size: 1rem; font-weight: normal;">Confidence: {confidence:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.success("✅ The model predicts a **low risk** of heart disease. Keep up the healthy lifestyle!")

        # Show patient details
        with st.expander("📋 Patient Input Details"):
            st.json(patient)

# ─── Footer ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Made with ❤️ using Streamlit | AI/ML Student Project | "
    "<a href='https://github.com'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
