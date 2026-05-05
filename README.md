# 🛡️ MediScan AI: Premium Multi-Disease Prediction Platform

MediScan AI is a state-of-the-art medical diagnostic assistance tool that leverages advanced Machine Learning to predict multiple health conditions. Featuring a premium **Glassmorphism UI** and an integrated **AI Symptom Chatbot**, it provides a comprehensive screening experience.

![Premium UI Dashboard](https://img.shields.io/badge/UI-Premium_Glassmorphism-blueviolet)
![ML Models](https://img.shields.io/badge/ML-XGBoost_%7C_RandomForest_%7C_LGBM-blue)
![Python](https://img.shields.io/badge/Python-3.9+-blue)

---

## ✨ Key Features

- **🚀 Multi-Disease Engine**: Automated training and prediction for binary and multi-class disease datasets.
- **💎 Glassmorphism UI**: High-end aesthetic with dark mode, interactive gauges, and smooth animations.
- **💬 AI Symptom Chatbot**: Natural language processing to analyze symptoms and suggest specific screenings.
- **📊 Interactive Analytics**: Real-time visualization of model accuracy, confusion matrices, and feature importances.
- **📥 Auto-Dataset Detection**: Simply drop a CSV into the data folder, and the AI handles the rest.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Custom CSS/JS)
- **Visualization**: Plotly Express, Graph Objects
- **ML Core**: Scikit-Learn, XGBoost, LightGBM
- **Data**: Pandas, NumPy
- **Optimization**: SMOTE (Imbalance Handling), GridSearchCV (Hyperparameter Tuning)

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Installation
```bash
# Clone the repository
git clone <repository-url>
cd disease_project

# Install dependencies
pip install -r requirements.txt
```

### 3. Training the AI
MediScan AI features an automated pipeline. To train all models at once:
```bash
python src/train_all.py
```

### 4. Launch the Platform
```bash
streamlit run app.py
```

---

## 📂 Dataset Architecture

MediScan AI is designed to be plug-and-play. Place your datasets in `data/raw/` to get started.

| Category | Dataset Name | Target Type | Status |
| :--- | :--- | :--- | :--- |
| **General** | `disease_dataset.csv` | Multi-class (15 Diseases) | ✅ Active |
| **Clinical** | `heart.csv` | Binary (Heart Disease) | ✅ Active |
| **Symptomatic** | `Disease_symptom_dataset.csv` | Binary (Patient Profile) | ✅ Active |
| **Specialized** | Parkinson's, Diabetes, Kidney | Customizable | 🔄 Ready |

### Adding Custom Datasets
The system will automatically detect new datasets if:
1. They are placed in `data/raw/`.
2. The target column matches keywords like `target`, `Outcome`, `disease`, or `status`.
3. Configuration is optionally added to `config.py` for custom icons and column dropping.

---

## 🧪 Model Performance

The platform evaluates multiple architectures for every disease:
- **XGBoost** & **LightGBM** (High performance)
- **Random Forest** (Robustness)
- **Logistic Regression** (Baseline)

Detailed metrics (Accuracy, F1-Score, ROC-AUC) are available in the **Model Insights** tab of the application.

---

## ⚖️ Medical Disclaimer

> [!WARNING]
> This software is for **educational and research purposes only**. It is not intended for use in clinical diagnosis or as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---
<p align="center">Designed with ❤️ for a Healthier World</p>
