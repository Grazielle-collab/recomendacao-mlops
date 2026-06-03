import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
import os

class UserMeanBaseline:
    """Modelo baseline: prediz a média de rating do usuário"""
    
    def __init__(self):
        self.user_means = {}
        self.global_mean = 0.0
        
    def fit(self, ratings_df):
        """Treina o modelo calculando a média por usuário"""
        self.user_means = ratings_df.groupby('user_id')['rating'].mean().to_dict()
        self.global_mean = ratings_df['rating'].mean()
        print(f"Modelo treinado com {len(self.user_means)} usuários")
        print(f"Média global: {self.global_mean:.3f}")
        
    def predict(self, user_id, item_id=None):
        """Prediz o rating para um usuário"""
        return self.user_means.get(user_id, self.global_mean)
    
    def predict_batch(self, users_df):
        """Prediz para múltiplos usuários"""
        return users_df.apply(lambda x: self.predict(x))
    
    def evaluate(self, test_df):
        """Avalia o modelo usando RMSE"""
        predictions = test_df['user_id'].apply(lambda x: self.predict(x))
        rmse = np.sqrt(mean_squared_error(test_df['rating'], predictions))
        return rmse

def train_model():
    """Função principal de treinamento"""
    
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
