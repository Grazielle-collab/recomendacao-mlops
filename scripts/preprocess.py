import pandas as pd
import json
import os

def preprocess():
    print("Carregando dados...")
    df = pd.read_csv("data/raw/ratings.csv")
    
    stats = {
        "total_ratings": len(df),
        "n_users": int(df.user_id.nunique()),
        "n_items": int(df.item_id.nunique()),
        "mean_rating": float(df.rating.mean()),
        "std_rating": float(df.rating.std())
    }
    
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    df.to_csv("data/processed/ratings_clean.csv", index=False)
    
    with open("metrics/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Preprocessamento concluido!")
    print(f"Shape: {df.shape}")
    print(f"Media rating: {stats['mean_rating']:.2f}")

if __name__ == "__main__":
    preprocess()
