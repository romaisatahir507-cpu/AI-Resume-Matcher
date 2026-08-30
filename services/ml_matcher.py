import os
import joblib


MODEL_PATH = "ml/resume_match_model.pkl"

def predict_ml_match(
    tfidf_score, 
    semantic_score,
    skills_match,
    experience_match,
    education_match,
    missing_skills_ratio
):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "ML model not found."
            "Please train the model first."
        )

    model = joblib.load(MODEL_PATH)

    features = [[
        tfidf_score,
        semantic_score,
        skills_match, 
        experience_match,
        education_match,
        missing_skills_ratio
    ]]


    prediction = model.predict(features)[0]

    # Keep score between 0 and 100
    prediction = max(0, min(100, prediction))

    return round(prediction, 2)