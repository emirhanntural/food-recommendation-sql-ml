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


def get_recent_click_ids(engine, user_id, n_clicks):
    clicks = pd.read_sql(
        text("""
            SELECT recipe_id
            FROM events
            WHERE user_id=:uid AND event_type='click'
            ORDER BY created_at DESC
            LIMIT :n
        """),
        engine,
        params={"uid": user_id, "n": n_clicks},
    )
    return clicks["recipe_id"].astype(int).tolist()


def make_weights(n, decay=0.6):
    # newest first: [1.0, decay, decay^2, ...]
    w = np.array([decay ** i for i in range(n)], dtype=float)
    w = w / (w.sum() + 1e-12)
    return w


def main(user_id=1, top_k=10, n_clicks=5, decay=0.6):
    engine = create_engine(DB_URL)

    # Recent clicks (newest first)
    seed_ids = get_recent_click_ids(engine, user_id, n_clicks)
    if len(seed_ids) == 0:
        print("No clicks yet.")
        return

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
    shown_set = set(shown["recipe_id"].astype(int)) if not shown.empty else set()

    # Numeric cleanup
    df["prep_time_min"] = pd.to_numeric(df["prep_time_min"], errors="coerce").fillna(df["prep_time_min"].median())
    df["health_score"] = pd.to_numeric(df["health_score"], errors="coerce").fillna(df["health_score"].median())

    # Feature matrix
    feat_text = make_feature_text(df)
    X_cat = feat_text.str.get_dummies(sep=" ")
    X_cat["z_prep_time"] = zscore(df["prep_time_min"])
    X_cat["z_health"] = zscore(df["health_score"])

    X = X_cat.values
    recipe_ids = df["recipe_id"].astype(int).to_numpy()
    idx_map = {rid: i for i, rid in enumerate(recipe_ids.tolist())}

    # Keep only clicks that exist in recipes
    seed_ids = [rid for rid in seed_ids if rid in idx_map]
    if len(seed_ids) == 0:
        print("Clicks not found in recipes.")
        return

    # Weighted user/session vector
    weights = make_weights(len(seed_ids), decay=decay)  # newest has highest weight
    seed_mat = np.vstack([X[idx_map[rid]] for rid in seed_ids])  # shape: (k, d)
    user_vec = (weights[:, None] * seed_mat).sum(axis=0, keepdims=True)  # shape: (1, d)

    sims = cosine_similarity(user_vec, X).ravel()
    df["similarity"] = sims

    # Filter out clicked items + recently shown
    recs = df.copy()
    recs = recs[~recs["recipe_id"].astype(int).isin(set(seed_ids))]
    recs = recs[~recs["recipe_id"].astype(int).isin(shown_set)]
    recs = recs.sort_values(["similarity", "health_score"], ascending=False).head(top_k)

    # Output
    last_seed_title = df.loc[df["recipe_id"].astype(int) == seed_ids[0], "title"].iloc[0]
    print(f"\nSeeds (newest→oldest): {seed_ids}")
    print(f"Most recent click: {last_seed_title}\n")
    print(recs[["recipe_id", "title", "similarity"]].to_string(index=False))

    # Log impressions
    now = datetime.utcnow()
    rows = [
        {"user_id": user_id, "recipe_id": int(rid), "event_type": "impression", "created_at": now}
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
    # n_clicks: kaç son click'i kullanacağımız
    # decay: 0.6 -> daha eskilere hızlı düşen ağırlık
    main(user_id=1, top_k=10, n_clicks=5, decay=0.6)
