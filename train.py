import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

# Adicionar src ao path para importar o modelo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baseline_model import UserMeanBaseline

def train_model():
    """Funcao principal de treinamento"""
    
    # Carregar dados
    print("Carregando dados...")
    df = pd.read_csv("data/ratings.csv")
    
    # Split treino/teste
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Treino: {len(train_df)} amostras")
    print(f"Teste: {len(test_df)} amostras")
    
    # Treinar modelo
    print("\nTreinando modelo baseline...")
    model = UserMeanBaseline()
    model.fit(train_df)
    
    # Avaliar
    rmse = model.evaluate(test_df)
    print(f"\nRMSE no teste: {rmse:.4f}")
    
    # Salvar modelo
    os.makedirs("models", exist_ok=True)
    with open("models/baseline.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Modelo salvo em models/baseline.pkl")
    
    return model, rmse

if __name__ == "__main__":
    train_model()
