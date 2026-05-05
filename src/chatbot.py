from typing import List, Dict


SYMPTOM_DISEASE_MAP = {
    # Diabetes
    "frequent urination": {"Diabetes": 0.9, "Kidney Disease": 0.5},
    "excessive thirst": {"Diabetes": 0.9},
    "unexplained weight loss": {"Diabetes": 0.7, "Liver Disease": 0.4},
    "blurred vision": {"Diabetes": 0.8},
    "slow healing wounds": {"Diabetes": 0.8},
    "tingling hands": {"Diabetes": 0.7, "Parkinson's Disease": 0.3},
    "fatigue": {"Diabetes": 0.6, "Heart Disease": 0.5,
                "Liver Disease": 0.5, "Kidney Disease": 0.6},
    "high blood sugar": {"Diabetes": 1.0},
    "glucose": {"Diabetes": 0.9},
    "insulin": {"Diabetes": 0.9},

    # Heart Disease
    "chest pain": {"Heart Disease": 0.95},
    "shortness of breath": {"Heart Disease": 0.85, "Kidney Disease": 0.4},
    "heart palpitations": {"Heart Disease": 0.85},
    "arm pain": {"Heart Disease": 0.8},
    "jaw pain": {"Heart Disease": 0.75},
    "dizziness": {"Heart Disease": 0.6, "Parkinson's Disease": 0.4},
    "high blood pressure": {"Heart Disease": 0.7, "Kidney Disease": 0.6},
    "irregular heartbeat": {"Heart Disease": 0.9},
    "sweating": {"Heart Disease": 0.5, "Diabetes": 0.4},
    "cholesterol": {"Heart Disease": 0.7},

    # Parkinson's Disease
    "tremors": {"Parkinson's Disease": 0.95},
    "shaking": {"Parkinson's Disease": 0.9},
    "stiffness": {"Parkinson's Disease": 0.85},
    "slow movement": {"Parkinson's Disease": 0.9},
    "balance problems": {"Parkinson's Disease": 0.8},
    "small handwriting": {"Parkinson's Disease": 0.85},
    "soft voice": {"Parkinson's Disease": 0.75},
    "masked face": {"Parkinson's Disease": 0.7},
    "freezing": {"Parkinson's Disease": 0.8},
    "bradykinesia": {"Parkinson's Disease": 0.9},

    # Liver Disease
    "jaundice": {"Liver Disease": 0.95},
    "yellow skin": {"Liver Disease": 0.95},
    "abdominal pain": {"Liver Disease": 0.7, "Kidney Disease": 0.5},
    "nausea": {"Liver Disease": 0.6, "Kidney Disease": 0.4},
    "vomiting": {"Liver Disease": 0.6},
    "dark urine": {"Liver Disease": 0.8, "Kidney Disease": 0.6},
    "swollen abdomen": {"Liver Disease": 0.8},
    "itchy skin": {"Liver Disease": 0.6, "Kidney Disease": 0.5},
    "loss of appetite": {"Liver Disease": 0.5, "Kidney Disease": 0.4},
    "pale stool": {"Liver Disease": 0.8},

    # Kidney Disease
    "swelling": {"Kidney Disease": 0.8},
    "foamy urine": {"Kidney Disease": 0.85},
    "back pain": {"Kidney Disease": 0.65},
    "reduced urination": {"Kidney Disease": 0.8},
    "blood in urine": {"Kidney Disease": 0.85},
    "creatinine": {"Kidney Disease": 0.9},
    "edema": {"Kidney Disease": 0.75},
    "anemia": {"Kidney Disease": 0.6, "Liver Disease": 0.4},
    "muscle cramps": {"Kidney Disease": 0.6},
    "confusion": {"Kidney Disease": 0.5, "Parkinson's Disease": 0.3},

    # General Disease Predictor
    "fever": {"General Disease Predictor": 0.5, "Patient Profile & Symptoms": 0.4},
    "cough": {"General Disease Predictor": 0.5, "Patient Profile & Symptoms": 0.4},
    "headache": {"General Disease Predictor": 0.4},
    "nausea": {"General Disease Predictor": 0.4, "Liver Disease": 0.4},
    "vomiting": {"General Disease Predictor": 0.4},
    "diarrhea": {"General Disease Predictor": 0.4},
    "skin rash": {"General Disease Predictor": 0.6},
    "itching": {"General Disease Predictor": 0.5},
    "runny nose": {"General Disease Predictor": 0.7},
    "sneezing": {"General Disease Predictor": 0.7},
    "sore throat": {"General Disease Predictor": 0.6},
    "chills": {"General Disease Predictor": 0.5},
    "body ache": {"General Disease Predictor": 0.4},
}

URGENCY_KEYWORDS = {
    "severe": "Seek immediate medical care",
    "sudden": "Seek immediate medical care",
    "intense": "Seek immediate medical care",
    "emergency": "Go to emergency room now",
    "unbearable": "Seek immediate medical care",
    "unconscious": "Call emergency services now",
    "cant breathe": "Seek immediate medical care",
    "can't breathe": "Seek immediate medical care",
}


def analyze_symptoms(text: str) -> Dict:
    """
    Analyze a free-text symptom description.
    Returns detected symptoms, possible diseases, urgency, recommendation.
    """
    text_lower = text.lower()

    # Check urgency
    urgency = "Consult a doctor soon"
    for kw, msg in URGENCY_KEYWORDS.items():
        if kw in text_lower:
            urgency = msg
            break

    # Match symptoms
    detected_symptoms = []
    disease_scores: Dict[str, float] = {}

    for symptom, disease_weights in SYMPTOM_DISEASE_MAP.items():
        # Check if any word in the symptom phrase appears in text
        symptom_words = symptom.split()
        if any(word in text_lower for word in symptom_words) or symptom in text_lower:
            detected_symptoms.append(symptom)
            for disease, weight in disease_weights.items():
                disease_scores[disease] = disease_scores.get(disease, 0) + weight

    # Normalize scores to 0-100%
    if disease_scores:
        max_score = max(disease_scores.values())
        disease_results = [
            {
                "name": d,
                "score": round((s / max_score) * 100, 1),
                "raw_score": s
            }
            for d, s in sorted(disease_scores.items(),
                                key=lambda x: x[1], reverse=True)
        ]
    else:
        disease_results = []

    # Generate recommendation
    if not detected_symptoms:
        recommendation = (
            "I couldn't identify specific symptoms in your description. "
            "Please describe your symptoms in more detail, e.g., "
            "'I have chest pain and shortness of breath'."
        )
    elif disease_results and disease_results[0]["score"] >= 70:
        top = disease_results[0]["name"]
        recommendation = (
            f"Your symptoms strongly suggest possible {top}. "
            "Please consult a healthcare professional for a proper diagnosis."
        )
    elif disease_results:
        top = ", ".join([d["name"] for d in disease_results[:2]])
        recommendation = (
            f"Your symptoms may be related to {top}. "
            "It is recommended to see a doctor for evaluation."
        )
    else:
        recommendation = "Please consult a doctor for a proper evaluation."

    return {
        "detected_symptoms": detected_symptoms,
        "possible_diseases": disease_results[:3],  # Top 3
        "urgency": urgency,
        "recommendation": recommendation,
        "symptom_count": len(detected_symptoms)
    }


def format_bot_response(analysis: Dict) -> str:
    """Format analysis result as a readable chatbot message."""
    lines = []

    if not analysis["detected_symptoms"]:
        return (
            "I couldn't identify specific symptoms from your description. "
            "Try describing symptoms like: 'I have chest pain, fatigue, and shortness of breath'."
        )

    lines.append(f"**Symptoms detected:** {', '.join(analysis['detected_symptoms'])}\n")

    if analysis["possible_diseases"]:
        lines.append("**Possible conditions:**")
        for d in analysis["possible_diseases"]:
            bar = "█" * int(d["score"] / 10) + "░" * (10 - int(d["score"] / 10))
            lines.append(f"  • {d['name']}: {bar} {d['score']}%")
        lines.append("")

    lines.append(f"**Urgency:** {analysis['urgency']}")
    lines.append(f"\n{analysis['recommendation']}")
    lines.append("\n⚠️ *This is for educational purposes only. Always consult a qualified doctor.*")

    return "\n".join(lines)


if __name__ == "__main__":
    test = "I have been having chest pain and shortness of breath with fatigue"
    result = analyze_symptoms(test)
    print(format_bot_response(result))
