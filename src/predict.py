import os
import json
import joblib
import numpy as np
import pandas as pd
import glob

MODEL_DIR = "models"


def list_trained_diseases():
    """Return list of trained disease names."""
    metric_files = glob.glob(f"{MODEL_DIR}/*_metrics.json")
    diseases = []
    for f in metric_files:
        with open(f) as fh:
            meta = json.load(fh)
        diseases.append({
            "display_name": meta["display_name"],
            "safe_name": meta["safe_name"],
            "icon": meta.get("icon", "🏥"),
            "accuracy": meta["accuracy"],
            "roc_auc": meta["roc_auc"],
            "best_model": meta["best_model"],
        })
    return diseases


def load_model(safe_name: str):
    """Load model, scaler, and metadata for a disease."""
    model_path = f"{MODEL_DIR}/{safe_name}_model.pkl"
    scaler_path = f"{MODEL_DIR}/{safe_name}_scaler.pkl"
    meta_path = f"{MODEL_DIR}/{safe_name}_metrics.json"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
        raise FileNotFoundError(
            f"Model files not found for '{safe_name}'. "
            "Run train_all.py first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        meta = json.load(f)

    return model, scaler, meta


def get_risk_label(prob: float):
    if prob >= 0.65:
        return "High", "danger"
    elif prob >= 0.35:
        return "Medium", "warning"
    return "Low", "success"


def get_health_recommendations(disease_name: str, risk: str):
    recs = {
        "Diabetes": {
            "High": [
                "Consult an endocrinologist immediately",
                "Adopt a low-sugar, high-fiber diet",
                "Exercise at least 30 min/day",
                "Monitor blood sugar weekly"
            ],
            "Medium": [
                "Schedule a glucose tolerance test",
                "Reduce refined carbohydrates and sugary drinks",
                "Aim for 20-30 min of daily walking"
            ],
            "Low": [
                "Maintain healthy BMI and stay active",
                "Annual glucose checkup recommended"
            ]
        },
        "Heart Disease": {
            "High": [
                "Seek immediate cardiology consultation",
                "Avoid strenuous physical activity until cleared",
                "Take prescribed medications regularly",
                "Monitor blood pressure daily"
            ],
            "Medium": [
                "Schedule a cardiac stress test",
                "Follow a heart-healthy, low-sodium diet",
                "Quit smoking and limit alcohol"
            ],
            "Low": [
                "Maintain regular exercise (cardio 3x/week)",
                "Annual cholesterol and BP checkup"
            ]
        },
        "Parkinson's Disease": {
            "High": [
                "Consult a neurologist immediately",
                "Discuss medication options with your doctor",
                "Consider physical and occupational therapy"
            ],
            "Medium": [
                "Schedule a neurological evaluation",
                "Engage in regular balance and coordination exercises"
            ],
            "Low": [
                "Maintain physical fitness and mental activity",
                "Report any new tremors or stiffness to your doctor"
            ]
        },
        "Liver Disease": {
            "High": [
                "See a hepatologist immediately",
                "Avoid alcohol completely",
                "Follow a liver-friendly diet (low-fat, avoid processed foods)"
            ],
            "Medium": [
                "Get liver function tests done",
                "Reduce alcohol and fatty food intake",
                "Stay well hydrated"
            ],
            "Low": [
                "Limit alcohol consumption",
                "Annual liver function test recommended"
            ]
        },
        "Kidney Disease": {
            "High": [
                "Consult a nephrologist immediately",
                "Follow a low-protein, low-sodium diet",
                "Monitor fluid intake and blood pressure"
            ],
            "Medium": [
                "Get kidney function tests (creatinine, eGFR)",
                "Stay well hydrated and reduce salt intake"
            ],
            "Low": [
                "Stay hydrated — drink 8 glasses of water daily",
                "Annual kidney function checkup recommended"
            ]
        }
    }

    # Generic fallback
    generic = {
        "High": ["Consult a specialist immediately",
                 "Follow medical advice and take prescribed medications"],
        "Medium": ["Schedule a doctor's appointment soon",
                   "Make lifestyle improvements"],
        "Low": ["Maintain a healthy lifestyle",
                "Annual health checkup recommended"]
    }

    disease_recs = recs.get(disease_name, generic)
    return disease_recs.get(risk, generic.get(risk, []))


def predict_single(safe_name: str, input_dict: dict):
    """
    Predict disease risk for a single patient.
    Returns: dict with prediction, probability, risk, recommendations
    """
    model, scaler, meta = load_model(safe_name)
    features = meta["features"]
    feature_stats = meta.get("feature_stats", {})

    # Build row using defaults for missing features
    row = {}
    for feat in features:
        if feat in input_dict:
            row[feat] = input_dict[feat]
        elif feat in feature_stats:
            row[feat] = feature_stats[feat]["mean"]
        else:
            row[feat] = 0.0

    df_row = pd.DataFrame([row])[features]
    row_scaled = scaler.transform(df_row)
    
    is_multiclass = meta.get("is_multiclass", False)
    label_map = meta.get("label_map")
    
    preds_proba = model.predict_proba(row_scaled)[0]
    pred_idx = int(model.predict(row_scaled)[0])
    
    if is_multiclass:
        prob = float(preds_proba[pred_idx])
        prediction_label = label_map.get(str(pred_idx), f"Class {pred_idx}") if label_map else str(pred_idx)
        
        # Top 3 predictions
        top_indices = np.argsort(preds_proba)[::-1][:3]
        top_classes = []
        if label_map:
            for idx in top_indices:
                top_classes.append({
                    "label": label_map.get(str(idx), str(idx)),
                    "prob": round(float(preds_proba[idx]) * 100, 1)
                })
        
        risk, severity = get_risk_label(prob)
        return {
            "prediction": pred_idx,
            "prediction_label": prediction_label,
            "probability": round(prob * 100, 1),
            "risk_level": risk,
            "severity": severity,
            "disease": meta["display_name"],
            "top_classes": top_classes,
            "is_multiclass": True,
            "message": f"Most likely condition: {prediction_label} ({round(prob * 100, 1)}%)",
            "recommendations": get_health_recommendations(prediction_label, risk)
        }
    else:
        prob = float(preds_proba[1])
        pred = pred_idx
        risk, severity = get_risk_label(prob)
        
        return {
            "prediction": pred,
            "probability": round(prob * 100, 1),
            "risk_level": risk,
            "severity": severity,
            "disease": meta["display_name"],
            "is_multiclass": False,
            "message": (
                f"High risk of {meta['display_name']} detected."
                if pred == 1 else
                f"Low risk of {meta['display_name']}."
            ),
            "recommendations": get_health_recommendations(meta["display_name"], risk)
        }


if __name__ == "__main__":
    diseases = list_trained_diseases()
    if not diseases:
        print("No trained models found. Run src/train_all.py first.")
    else:
        print(f"Trained diseases: {[d['display_name'] for d in diseases]}")
