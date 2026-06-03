# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import Optional, List
import os
import sys

# Adicionar src ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar a classe do modelo
from src.models.baseline_model import UserMeanBaseline

app = FastAPI(title="Sistema de Recomendacao MLOps",
              description="API para predicao de ratings e recomendacoes",
              version="1.0.0")

class RatingRequest(BaseModel):
    user_id: int
    item_id: Optional[int] = None

class RatingResponse(BaseModel):
    user_id: int
    predicted_rating: float

class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[int]

model = None
item_popularity = None

@app.on_event("startup")
async def load_model():
    global model, item_popularity
    
    model_path = "models/baseline.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Modelo carregado com sucesso")
    else:
        print("Modelo nao encontrado. Execute treinamento primeiro.")
    
    ratings_path = "data/ratings.csv"
    if os.path.exists(ratings_path):
        ratings = pd.read_csv(ratings_path)
        item_popularity = ratings.groupby('item_id')['rating'].agg(['mean', 'count'])
        item_popularity['score'] = item_popularity['mean'] * np.log1p(item_popularity['count'])
        item_popularity = item_popularity.sort_values('score', ascending=False)
        print(f"{len(item_popularity)} itens carregados para recomendacao")

@app.get("/")
async def root():
    return {
        "message": "Sistema de Recomendacao MLOps",
        "status": "online",
        "endpoints": ["/health", "/predict", "/recommend"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "items_loaded": item_popularity is not None
    }

@app.post("/predict", response_model=RatingResponse)
async def predict(request: RatingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo nao carregado")
    
    try:
        pred = model.predict(request.user_id, request.item_id)
        return RatingResponse(user_id=request.user_id, predicted_rating=float(pred))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    if item_popularity is None:
        raise HTTPException(status_code=503, detail="Dados nao carregados")
    
    try:
        top_items = item_popularity.head(request.top_k).index.tolist()
        return RecommendResponse(
            user_id=request.user_id,
            recommendations=top_items
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
