from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI(title="E-Commerce Recommendation API")

model = tf.keras.models.load_model("ncf_model.keras")

user_enc = joblib.load("user_encoder.pkl")
item_enc = joblib.load("item_encoder.pkl")
category_enc = joblib.load("category_encoder.pkl")
fit_enc = joblib.load("fit_encoder.pkl")

class RecommendationRequest(BaseModel):
    user_id: str
    item_ids: list
    category: list
    fit: list

@app.post("/predict/")
def recommend(data: RecommendationRequest):
    try:
        def safe_transform(encoder, values):
            return [
                encoder.transform([v])[0] if v in encoder.classes_ else -1
                for v in values
            ]

        user_encoded = safe_transform(user_enc, [data.user_id])[0]
        item_encoded = safe_transform(item_enc, data.item_ids)
        category_encoded = safe_transform(category_enc, data.category)
        fit_encoded = safe_transform(fit_enc, data.fit)

        valid_idx = [i for i, v in enumerate(item_encoded) if v != -1]
        if not valid_idx:
            raise HTTPException(status_code=400, detail="No valid items found for given IDs")

        user_input = np.array([user_encoded] * len(valid_idx))
        item_input = np.array([item_encoded[i] for i in valid_idx])
        category_input = np.array([category_encoded[i] for i in valid_idx])
        fit_input = np.array([fit_encoded[i] for i in valid_idx])

        preds = model.predict([user_input, item_input, category_input, fit_input]).flatten()

        top_indices = np.argsort(-preds)
        top_items = [data.item_ids[i] for i in valid_idx[:5]]

        return {
            "user_id": data.user_id,
            "recommended_items": top_items,
            "scores": preds[top_indices[:5]].tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/")
def home():
    return {"message": "E-Commerce Recommendation API is running"}