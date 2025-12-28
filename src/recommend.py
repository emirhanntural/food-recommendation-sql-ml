import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/foodrec")

def make_feature_text(df: pd.DataFrame) -> pd.Series:
    def clean(x):
        if x is None:
            return ""
        return str(x).strip().lower().replace(" ", "_")

    return (
        "main_" + df["main_ingredient"].map(clean) + " "
        "cuisine_" + df["cuisine"].map(clean) + " "
        "cat_" + df["category"].map(clean) + " "
        "diff_" + df["difficulty"].map(clean)
    )

def main(user_id: int = 1, top_k: int = 10):
    engine = create_engine(DB_URL)

    # 1) Son tıklanan recipe (seed)
    seed = pd.read_sql(
        text("""
        SELECT recipe_id
        FROM events
        WHERE user_id=:uid AND event_type='click'
        ORDER BY created_at DESC
        LIMIT 1;
        """),
        engine,
        params={"uid": user_id}
    )

    if seed.empty:
        print("No clicks found for this user yet.")
        return

    seed_id = int(seed.loc[0, "recipe_id"])

    df = pd.read_sql(
        text("""
        SELECT
          r.recipe_id,
          r.title,
          r.prep_time_min,
          r.category,
          r.cuisine,
          r.health_score,
          r.difficulty,
          i.name AS main_ingredient
        FROM recipes r
        LEFT JOIN recipe_ingredients ri
          ON ri.recipe_id = r.recipe_id AND ri.is_main = TRUE
        LEFT JOIN ingredients i
          ON i.ingredient_id = ri.ingredient_id;
        """),
        engine
    )

    shown = pd.read_sql(
        text("""
        SELECT recipe_id
        FROM events
        WHERE user_id=:uid AND event_type='impression'
        ORDER BY created_at DESC
        LIMIT 20;
        """),
        engine,
        params={"uid": user_id}
    )
    shown_set = set(shown["recipe_id"].astype(int).tolist())

    # 4) Feature engineering
    df["prep_time_min"] = pd.to_numeric(df["prep_time_min"], errors="coerce").fillna(df["prep_time_min"].median())
    df["health_score"] = pd.to_numeric(df["health_score"], errors="coerce").fillna(df["health_score"].median())

    # Text-like categorical features (main_ingredient/cuisine/category/difficulty)
    feat_text = make_feature_text(df)

    # Simple vectorization: one-hot via pandas get_dummies on tokens
    # (No heavy NLP yet; portfolio-friendly and fast)
    tokens = feat_text.str.get_dummies(sep=" ")

    # Add scaled numeric features
    def z(x):
        x = x.astype(float)
        return (x - x.mean()) / (x.std() + 1e-9)

    tokens["z_prep_time"] = z(df["prep_time_min"])
    tokens["z_health"] = z(df["health_score"])

    # 5) Similarity
    X = tokens.values
    idx_map = {rid: i for i, rid in enumerate(df["recipe_id"].astype(int).tolist())}

    if seed_id not in idx_map:
        print("Seed recipe not found in recipes table.")
        return

    seed_idx = idx_map[seed_id]
    sims = cosine_similarity(X[seed_idx:seed_idx+1], X).ravel()

    df["similarity"] = sims

    # 6) Filter: remove seed + recently shown
    recs = df[df["recipe_id"].astype(int) != seed_id].copy()
    recs = recs[~recs["recipe_id"].astype(int).isin(shown_set)].copy()

    # 7) Top-K
    recs = recs.sort_values(["similarity", "health_score"], ascending=[False, False]).head(top_k)

    # Print results
    seed_title = df.loc[df["recipe_id"] == seed_id, "title"].iloc[0]
    print(f"\nSeed (last clicked): {seed_id} — {seed_title}\n")
    print(recs[["recipe_id","title","cuisine","category","main_ingredient","prep_time_min","health_score","similarity"]].to_string(index=False))

if __name__ == "__main__":
    # default user_id=1
    main(user_id=1, top_k=10)
