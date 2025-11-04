from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import traceback
import random

app = FastAPI(title="E-Commerce Recommendation API")

model = tf.keras.models.load_model("ncf_model.keras")
user_enc = joblib.load("user_encoder.pkl")
item_enc = joblib.load("item_encoder.pkl")
category_enc = joblib.load("category_encoder.pkl")
fit_enc = joblib.load("fit_encoder.pkl")

class RecommendationRequest(BaseModel):
    user_id: int
    item_ids: list
    category: list
    fit: list

def safe_encode_single(encoder, value):
    """Encode a single value, assign 'unknown' index for unseen ones."""
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return random.randint(0, len(encoder.classes_) - 1)

def safe_encode_multiple(encoder, values):
    """Encode a list safely."""
    encoded = []
    for v in values:
        if v in encoder.classes_:
            encoded.append(encoder.transform([v])[0])
        else:
            encoded.append(random.randint(0, len(encoder.classes_) - 1))
    return encoded

@app.post("/predict/")
def recommend(data: RecommendationRequest):
    try:
        user_encoded = safe_encode_single(user_enc, data.user_id)
        item_encoded = safe_encode_multiple(item_enc, data.item_ids)
        category_encoded = safe_encode_multiple(category_enc, data.category)
        fit_encoded = safe_encode_multiple(fit_enc, data.fit)

        n = len(item_encoded)
        category_encoded = (category_encoded * (n // len(category_encoded) + 1))[:n]
        fit_encoded = (fit_encoded * (n // len(fit_encoded) + 1))[:n]

        user_input = np.array([user_encoded] * n)
        item_input = np.array(item_encoded)
        category_input = np.array(category_encoded)
        fit_input = np.array(fit_encoded)

        preds = model.predict([user_input, item_input, category_input, fit_input]).flatten()

        top_indices = np.argsort(-preds)[:5]
        top_items = [data.item_ids[i] for i in top_indices]

        return {
            "user_id": data.user_id,
            "recommended_items": top_items,
            "scores": preds[top_indices].tolist(),
        }

    except Exception as e:
        print("Error Traceback:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def home():
    return {"message": "E-Commerce Recommendation API is running!"}