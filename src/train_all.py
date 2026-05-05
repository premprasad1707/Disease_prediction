import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import scan_datasets, MODEL_DIR

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE


def json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def preprocess_df(df, target_col, drop_cols):
    """Clean and encode a disease dataframe."""
    df = df.copy()

    # Drop unwanted columns
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Fix mixed-type columns
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > len(df) * 0.7:
                df[col] = converted

    # Encode binary Yes/No
    for col in df.columns:
        if df[col].dtype == object:
            unique = df[col].dropna().unique()
            if set(str(u).lower() for u in unique) <= {"yes", "no"}:
                df[col] = df[col].map({"yes": 1, "Yes": 1, "no": 0, "No": 0})

    # Drop rows where target is NaN
    df = df.dropna(subset=[target_col])

    # Encode target
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
    else:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna(subset=[target_col])
    
    if len(df) == 0:
        return df

    # Fill missing values for features
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

    # Encode remaining categoricals in features
    for col in df.select_dtypes(include="object").columns:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def train_disease(dataset_config):
    name = dataset_config["display_name"]
    safe_name = name.replace(" ", "_").replace("'", "")
    file = dataset_config["file"]
    target = dataset_config["target"]
    drop_cols = dataset_config["drop_cols"]

    print(f"\n{'='*55}")
    print(f"TRAINING: {name}")
    print(f"{'='*55}")

    df = pd.read_csv(file)
    print(f"[INFO] Shape: {df.shape} | Target: {target}")
    print(f"[INFO] Class balance: {df[target].value_counts().to_dict()}")

    df = preprocess_df(df, target, drop_cols)
    features = [c for c in df.columns if c != target]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SMOTE if imbalanced
    ratio = y_train.value_counts().max() / y_train.value_counts().min()
    if ratio > 2:
        print(f"[INFO] Applying SMOTE (imbalance ratio={ratio:.1f})")
        sm = SMOTE(random_state=42)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42,
                                  eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }

    results = []
    trained = {}

    is_multiclass = len(np.unique(y)) > 2
    
    for mname, model in models.items():
        try:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            
            if is_multiclass:
                proba = model.predict_proba(X_test_s)
                auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
            else:
                proba = model.predict_proba(X_test_s)[:, 1]
                auc = roc_auc_score(y_test, proba)
                
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            
            results.append({"model": mname, "accuracy": round(acc, 4),
                             "f1": round(f1, 4), "roc_auc": round(auc, 4)})
            trained[mname] = model
            print(f"  {mname:<25} Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")
        except Exception as e:
            print(f"  [SKIP] {mname}: {e}")

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    best_name = results_df.iloc[0]["model"]
    best_model = trained[best_name]
    print(f"\n[BEST] {best_name} (AUC={results_df.iloc[0]['roc_auc']})")

    # Tune best model
    if best_name == "Random Forest":
        param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10]}
    elif best_name in ["XGBoost", "LightGBM", "Gradient Boosting"]:
        param_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
    else:
        param_grid = {"C": [0.1, 1, 10]}

    try:
        gs = GridSearchCV(best_model, param_grid, cv=5,
                          scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train_s, y_train)
        best_model = gs.best_estimator_
    except Exception as e:
        print(f"[WARN] Tuning skipped: {e}")

    # Final evaluation
    final_preds = best_model.predict(X_test_s)
    final_proba = best_model.predict_proba(X_test_s)
    
    if is_multiclass:
        final_auc = roc_auc_score(y_test, final_proba, multi_class="ovr", average="macro")
    else:
        final_auc = roc_auc_score(y_test, final_proba[:, 1])
        
    final_acc = accuracy_score(y_test, final_preds)
    cm = confusion_matrix(y_test, final_preds).tolist()

    # Get label names if encoded
    label_map = None
    if df[target].dtype != np.number:
        # This was encoded by preprocess_df
        # We need to recover the mapping. Preprocess_df uses LabelEncoder.
        # Let's re-run it briefly to get the classes
        temp_le = LabelEncoder()
        temp_le.fit(pd.read_csv(file)[target].astype(str))
        label_map = {int(i): str(name) for i, name in enumerate(temp_le.classes_)}

    # Feature importance
    fi = None
    if hasattr(best_model, "feature_importances_"):
        fi = {str(k): float(v) for k, v in zip(features, best_model.feature_importances_)}
        fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    # Save everything
    joblib.dump(best_model, f"{MODEL_DIR}/{safe_name}_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/{safe_name}_scaler.pkl")

    # Get min/max for UI sliders
    stats = {}
    for col in features:
        orig_col = df[col]
        stats[col] = {
            "min": float(orig_col.min()),
            "max": float(orig_col.max()),
            "mean": float(orig_col.mean()),
            "unique": int(orig_col.nunique())
        }

    meta = {
        "display_name": name,
        "safe_name": safe_name,
        "icon": dataset_config.get("icon", "🏥"),
        "features": features,
        "target": target,
        "best_model": best_name,
        "accuracy": round(final_acc, 4),
        "roc_auc": round(final_auc, 4),
        "confusion_matrix": cm,
        "feature_importance": fi,
        "feature_stats": stats,
        "all_results": results,
        "label_map": label_map,
        "is_multiclass": is_multiclass
    }

    with open(f"{MODEL_DIR}/{safe_name}_metrics.json", "w") as f:
        json.dump(json_safe(meta), f, indent=2)
    with open(f"{MODEL_DIR}/{safe_name}_features.json", "w") as f:
        json.dump(features, f, indent=2)

    print(f"[SAVED] {safe_name}_model.pkl | Acc={final_acc:.3f} | AUC={final_auc:.3f}")
    return meta


def train_all():
    datasets = scan_datasets()
    if not datasets:
        print("[ERROR] No datasets found in data/raw/")
        print("  → Download datasets from Kaggle and place CSVs in data/raw/")
        return []

    all_metrics = []
    for ds in datasets:
        try:
            m = train_disease(ds)
            all_metrics.append(m)
        except Exception as e:
            print(f"[ERROR] Failed to train {ds['display_name']}: {e}")

    print(f"\n{'='*55}")
    print("TRAINING SUMMARY")
    print(f"{'='*55}")
    for m in all_metrics:
        print(f"  {m['display_name']:<25} "
              f"Best={m['best_model']:<25} "
              f"Acc={m['accuracy']:.3f} AUC={m['roc_auc']:.3f}")

    return all_metrics


if __name__ == "__main__":
    train_all()
