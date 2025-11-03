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
        if data.user_id in user_enc.classes_:
            user_encoded = user_enc.transform([data.user_id])[0]
        else:
            user_encoded = np.random.randint(0, len(user_enc.classes_))

        valid_items = [iid for iid in data.item_ids if iid in item_enc.classes_]
        if not valid_items:
            raise HTTPException(status_code=400, detail="No valid item_ids found for prediction.")
        item_encoded = item_enc.transform(valid_items)
        category_encoded = [
            category_enc.transform([c])[0] if c in category_enc.classes_ else 0
            for c in data.category
        ]
        fit_encoded = [
            fit_enc.transform([f])[0] if f in fit_enc.classes_ else 0
            for f in data.fit
        ]

        user_input = np.array([user_encoded] * len(item_encoded))
        item_input = np.array(item_encoded)
        category_input = np.array(category_encoded)
        fit_input = np.array(fit_encoded)

        preds = model.predict([user_input, item_input, category_input, fit_input]).flatten()

        top_indices = np.argsort(-preds)
        top_items = [valid_items[i] for i in top_indices[:5]]

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