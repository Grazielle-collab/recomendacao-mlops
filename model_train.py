import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os

class UserMeanBaseline:
    def __init__(self):
        self.user_means = {}
        self.global_mean = 0.0
    
    def fit(self, ratings_df):
        self.user_means = ratings_df.groupby('user_id')['rating'].mean().to_dict()
        self.global_mean = ratings_df['rating'].mean()
        print(f"Modelo treinado com {len(self.user_means)} usuarios")
        print(f"Media global: {self.global_mean:.3f}")
    
    def predict(self, user_id, item_id=None):
        return self.user_means.get(user_id, self.global_mean)
    
    def evaluate(self, test_df):
        predictions = test_df['user_id'].apply(lambda x: self.predict(x))
        rmse = np.sqrt(mean_squared_error(test_df['rating'], predictions))
        return rmse

def train_model():
    print("Carregando dados...")
    df = pd.read_csv("data/raw/ratings.csv")
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Treino: {len(train_df)} amostras")
    print(f"Teste: {len(test_df)} amostras")
    
    print("\nTreinando modelo baseline...")
    model = UserMeanBaseline()
    model.fit(train_df)
    
    rmse = model.evaluate(test_df)
    print(f"\nRMSE no teste: {rmse:.4f}")
    
    # Salvar como dicionário (evita problemas com classes)
    os.makedirs("models", exist_ok=True)
    model_data = {
        'user_means': model.user_means,
        'global_mean': model.global_mean
    }
    
    with open("models/baseline.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Modelo salvo em models/baseline.pkl")

if __name__ == "__main__":
    train_model()
