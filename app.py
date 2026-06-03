# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pickle
import pandas as pd
import numpy as np
from typing import Optional, List
import os

# Definir a classe do modelo (necessária para o pickle)
class UserMeanBaseline:
    def __init__(self):
        self.user_means = {}
        self.global_mean = 0.0
        
    def predict(self, user_id, item_id=None):
        return self.user_means.get(user_id, self.global_mean)

# Modelos de dados
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

# Variáveis globais
model = None
item_popularity = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, item_popularity
    
    print("Inicializando API...")
    
    # Carregar modelo
    model_path = "models/baseline.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("? Modelo carregado com sucesso")
    else:
        print("?? Modelo nao encontrado. Execute model_train.py primeiro")
    
    # Carregar dados para recomendacao
    ratings_path = "data/processed/ratings_clean.csv"
    if not os.path.exists(ratings_path):
        ratings_path = "data/raw/ratings.csv"
    
    if os.path.exists(ratings_path):
        ratings = pd.read_csv(ratings_path)
        item_popularity = ratings.groupby('item_id')['rating'].mean().sort_values(ascending=False)
        print(f"? {len(item_popularity)} itens carregados")
    else:
        print("?? Dados nao encontrados")
    
    yield  # Aqui a API roda
    
    # Shutdown
    print("Desligando API...")

# Criar app com lifespan
app = FastAPI(
    title="Sistema de Recomendacao MLOps",
    description="API para predicao de ratings e recomendacao de filmes",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "message": "Sistema de Recomendacao MLOps",
        "status": "online",
        "version": "2.0.0",
        "endpoints": ["/health", "/predict", "/recommend", "/docs"]
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
    
    pred = model.predict(request.user_id, request.item_id)
    return RatingResponse(user_id=request.user_id, predicted_rating=float(pred))

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    if item_popularity is None:
        raise HTTPException(status_code=503, detail="Dados nao carregados")
    
    top_items = item_popularity.head(request.top_k).index.tolist()
    return RecommendResponse(
        user_id=request.user_id,
        recommendations=top_items
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8001,
        reload=False
    )
