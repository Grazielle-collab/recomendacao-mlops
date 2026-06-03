import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class UserMeanBaseline:
    """Modelo baseline: prediz a media de rating do usuario"""
    
    def __init__(self):
        self.user_means = {}
        self.global_mean = 0.0
        
    def fit(self, ratings_df):
        """Treina o modelo calculando a media por usuario"""
        self.user_means = ratings_df.groupby('user_id')['rating'].mean().to_dict()
        self.global_mean = ratings_df['rating'].mean()
        print(f"Modelo treinado com {len(self.user_means)} usuarios")
        print(f"Media global: {self.global_mean:.3f}")
        
    def predict(self, user_id, item_id=None):
        """Prediz o rating para um usuario"""
        return self.user_means.get(user_id, self.global_mean)
    
    def evaluate(self, test_df):
        """Avalia o modelo usando RMSE"""
        predictions = test_df['user_id'].apply(lambda x: self.predict(x))
        rmse = np.sqrt(mean_squared_error(test_df['rating'], predictions))
        return rmse
