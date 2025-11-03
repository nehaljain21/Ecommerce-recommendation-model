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
        user_encoded = user_enc.transform([data.user_id])[0]
        item_encoded = item_enc.transform(data.item_ids)
        category_encoded = category_enc.transform(data.category)
        fit_encoded = fit_enc.transform(data.fit)

        user_input = np.array([user_encoded] * len(item_encoded))
        item_input = np.array(item_encoded)
        category_input = np.array(category_encoded)
        fit_input = np.array(fit_encoded)

        preds = model.predict([user_input, item_input, category_input, fit_input]).flatten()

        top_indices = np.argsort(-preds)
        top_items = [data.item_ids[i] for i in top_indices[:5]]
        return {
            "user_id": data.user_id,
            "recommended_items": top_items,
            "scores": preds[top_indices[:5]].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "E-Commerce Recommendation API is running"}