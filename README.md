# ❤️ Heart Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

<!-- Replace <your-username> with your actual GitHub username after deployment -->
<!-- [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-username>-heart-disease-ml-project.streamlit.app) -->

A comprehensive machine learning project that predicts the likelihood of heart disease using multiple classification algorithms. The project compares **6 different models**, performs hyperparameter tuning, and provides an **interactive web dashboard** built with Streamlit.

### 🌐 https://cp-heart-disease-ml-project.streamlit.app/

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Deployment](#deployment)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🔍 Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can significantly improve treatment outcomes. This project uses the **Cleveland Heart Disease dataset** to build and compare multiple ML classifiers.

### What This Project Demonstrates

- **Data Preprocessing** — Handling missing values, feature scaling, encoding
- **Exploratory Data Analysis (EDA)** — Statistical analysis and visualizations
- **Model Training & Evaluation** — 6 different algorithms with cross-validation
- **Hyperparameter Tuning** — GridSearchCV for optimal parameters
- **Model Comparison** — Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization** — Confusion matrices, ROC curves, feature importance
- **Interactive Web App** — Streamlit dashboard with real-time predictions
- **Cloud Deployment** — Deployed free on Streamlit Community Cloud

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) |
| **`target`** | **Heart disease (1 = disease, 0 = no disease)** |

---

## 📁 Project Structure

```
heart-disease-ml-project/
├── .streamlit/
│   └── config.toml          # Streamlit theme & server config
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Load and generate dataset
│   ├── preprocessing.py     # Data cleaning and feature engineering
│   ├── eda.py               # Exploratory Data Analysis & plots
│   ├── models.py            # Model training and evaluation
│   └── visualizations.py    # Result visualizations (ROC, CM, etc.)
├── data/
│   └── heart.csv            # Heart disease dataset
├── models/                  # Saved trained models
├── notebooks/
│   └── exploration.ipynb    # Jupyter notebook for interactive EDA
├── results/
│   └── plots/               # Generated visualization plots
├── app.py                   # 🌐 Streamlit Web App (main entry)
├── main.py                  # CLI pipeline entry point
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

```bash
# Clone the repository
git clone https://github.com/<userchandraprakash>/heart-disease-ml-project.git
cd heart-disease-ml-project

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Web Dashboard (Streamlit)

```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

### Run the CLI Pipeline

```bash
python main.py
```

### Use as a Module

```python
from src.data_loader import load_heart_data
from src.preprocessing import preprocess_data
from src.models import train_and_evaluate_all

df = load_heart_data()
X_train, X_test, y_train, y_test = preprocess_data(df)[:4]
results = train_and_evaluate_all(X_train, X_test, y_train, y_test)
```

---

## 🤖 Models Implemented

| # | Model | Description |
|---|-------|-------------|
| 1 | **Logistic Regression** | Linear model for binary classification |
| 2 | **K-Nearest Neighbors (KNN)** | Instance-based learning |
| 3 | **Support Vector Machine (SVM)** | Maximum margin classifier |
| 4 | **Decision Tree** | Tree-based rule learning |
| 5 | **Random Forest** | Ensemble of decision trees |
| 6 | **Gradient Boosting** | Sequential ensemble method |

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~95% | ~94% | ~97% | ~96% | ~99% |
| K-Nearest Neighbors | ~89% | ~86% | ~94% | ~90% | ~96% |
| Support Vector Machine | ~98% | ~97% | ~100% | ~99% | ~100% |
| Decision Tree | ~97% | ~97% | ~97% | ~97% | ~96% |
| Random Forest | ~97% | ~97% | ~97% | ~97% | ~100% |
| Gradient Boosting | ~97% | ~97% | ~97% | ~97% | ~100% |

> *Results may vary based on random state and data split.*

---

## 🚀 Deployment

### Deploy on Streamlit Community Cloud (Free!)

1. Push this project to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click **"New app"**
5. Select your repository, branch (`main`), and main file (`app.py`)
6. Click **Deploy!**

Your app will be live at: `https://<your-username>-heart-disease-ml-project.streamlit.app`

---

## 🧠 Key Learnings

1. **Feature Selection Matters** — `cp`, `thalach`, and `oldpeak` are strong predictors
2. **Ensemble Methods Excel** — Random Forest and Gradient Boosting consistently outperform single models
3. **Scaling is Critical** — SVM and KNN performance improved significantly with StandardScaler
4. **Cross-Validation** — Provides more reliable performance estimates than a single train/test split

---

## 🔮 Future Improvements

- [ ] Add Neural Network (MLP) classifier
- [ ] Implement SHAP for model explainability
- [ ] Add more datasets for generalizability
- [ ] Implement automated ML pipeline with MLflow
- [ ] Deploy model as a REST API

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit](https://streamlit.io/)
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)

---

<p align="center">
  Made with ❤️ for AI/ML learning | ⭐ Star this repo if you found it useful!
</p>
