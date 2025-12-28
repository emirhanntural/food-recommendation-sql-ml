import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/foodrec")


def make_feature_text(df):
    # Basic text features from categorical fields
    def clean(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return str(x).strip().lower().replace(" ", "_")

    return (
        "main_" + df["main_ingredient"].map(clean) + " "
        "cuisine_" + df["cuisine"].map(clean) + " "
        "cat_" + df["category"].map(clean) + " "
        "diff_" + df["difficulty"].map(clean)
    )


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def main(user_id=1, top_k=10):
    engine = create_engine(DB_URL)

    # Get last clicked recipe
    seed = pd.read_sql(
        text("""
            SELECT recipe_id
            FROM events
            WHERE user_id=:uid AND event_type='click'
            ORDER BY created_at DESC
            LIMIT 1
        """),
        engine,
        params={"uid": user_id}
    )

    if seed.empty:
        print("No clicks yet.")
        return

    seed_id = int(seed.loc[0, "recipe_id"])

    # Load recipes
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
              ON i.ingredient_id = ri.ingredient_id
        """),
        engine
    )

    # Last shown recipes
    shown = pd.read_sql(
        text("""
            SELECT recipe_id
            FROM events
            WHERE user_id=:uid AND event_type='impression'
            ORDER BY created_at DESC
            LIMIT 20
        """),
        engine,
        params={"uid": user_id}
    )
    shown_set = set(shown["recipe_id"].astype(int))

    # Numeric cleanup
    df["prep_time_min"] = pd.to_numeric(df["prep_time_min"], errors="coerce").fillna(df["prep_time_min"].median())
    df["health_score"] = pd.to_numeric(df["health_score"], errors="coerce").fillna(df["health_score"].median())

    # Feature matrix
    feat_text = make_feature_text(df)
    X_cat = feat_text.str.get_dummies(sep=" ")

    X_cat["z_prep_time"] = zscore(df["prep_time_min"])
    X_cat["z_health"] = zscore(df["health_score"])

    X = X_cat.values

    idx_map = {rid: i for i, rid in enumerate(df["recipe_id"].astype(int))}

    if seed_id not in idx_map:
        print("Seed not found.")
        return

    seed_idx = idx_map[seed_id]
    sims = cosine_similarity(X[seed_idx:seed_idx + 1], X).ravel()
    df["similarity"] = sims

    # Filter and rank
    recs = df[df["recipe_id"] != seed_id]
    recs = recs[~recs["recipe_id"].isin(shown_set)]
    recs = recs.sort_values(["similarity", "health_score"], ascending=False).head(top_k)

    # Output
    seed_title = df.loc[df["recipe_id"] == seed_id, "title"].iloc[0]
    print(f"\nSeed: {seed_title}\n")
    print(recs[["recipe_id", "title", "similarity"]].to_string(index=False))

    # Log impressions
    now = datetime.utcnow()
    rows = [
        {
            "user_id": user_id,
            "recipe_id": int(rid),
            "event_type": "impression",
            "created_at": now
        }
        for rid in recs["recipe_id"]
    ]

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO events (user_id, recipe_id, event_type, created_at)
                VALUES (:user_id, :recipe_id, :event_type, :created_at)
            """),
            rows
        )

    print(f"\n{len(rows)} impressions logged.")


if __name__ == "__main__":
    main()
