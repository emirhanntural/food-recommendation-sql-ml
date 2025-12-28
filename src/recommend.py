import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/foodrec")


def make_feature_text(df: pd.DataFrame) -> pd.Series:
    """Create a simple token string per recipe using categorical fields."""
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


def zscore(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    return (x - x.mean()) / (x.std() + 1e-9)


def get_seed_id(engine, user_id: int) -> int | None:
    seed = pd.read_sql(
        text("""
            SELECT recipe_id
            FROM events
            WHERE user_id=:uid AND event_type='click'
            ORDER BY created_at DESC
            LIMIT 1;
        """),
        engine,
        params={"uid": user_id},
    )
    if seed.empty:
        return None
    return int(seed.loc[0, "recipe_id"])


def get_recipes_df(engine) -> pd.DataFrame:
    return pd.read_sql(
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
        engine,
    )


def get_recent_impressions(engine, user_id: int, limit: int = 20) -> set[int]:
    shown = pd.read_sql(
        text("""
            SELECT recipe_id
            FROM events
            WHERE user_id=:uid AND event_type='impression'
            ORDER BY created_at DESC
            LIMIT :lim;
        """),
        engine,
        params={"uid": user_id, "lim": limit},
    )
    if shown.empty:
        return set()
    return set(shown["recipe_id"].astype(int).tolist())


def compute_recommendations(df: pd.DataFrame, seed_id: int, shown_set: set[int], top_k: int = 10) -> pd.DataFrame:
    # numeric cleanup
    df = df.copy()
    df["prep_time_min"] = pd.to_numeric(df["prep_time_min"], errors="coerce")
    df["health_score"] = pd.to_numeric(df["health_score"], errors="coerce")

    df["prep_time_min"] = df["prep_time_min"].fillna(df["prep_time_min"].median())
    df["health_score"] = df["health_score"].fillna(df["health_score"].median())

    # categorical tokens -> one-hot
    feat_text = make_feature_text(df)
    tokens = feat_text.str.get_dummies(sep=" ")

    # numeric features (scaled)
    tokens["z_prep_time"] = zscore(df["prep_time_min"])
    tokens["z_health"] = zscore(df["health_score"])

    X = tokens.values
    recipe_ids = df["recipe_id"].astype(int).tolist()
    idx_map = {rid: i for i, rid in enumerate(recipe_ids)}

    if seed_id not in idx_map:
        raise ValueError("Seed recipe not found in recipes table.")

    seed_idx = idx_map[seed_id]
    sims = cosine_similarity(X[seed_idx:seed_idx + 1], X).ravel()
    df["similarity"] = sims

    # filter out seed + recently shown
    recs = df[df["recipe_id"].astype(int) != seed_id].copy()
    if shown_set:
        recs = recs[~recs["recipe_id"].astype(int).isin(shown_set)].copy()

    # rank and return top_k
    recs = recs.sort_values(["similarity", "health_score"], ascending=[False, False]).head(top_k)
    return recs


def log_impressions(engine, user_id: int, recs: pd.DataFrame, session_id: str | None = None) -> int:
    """
    Insert recommendation results as impressions into events.
    Assumes events has at least: user_id, recipe_id, event_type, created_at
    Optionally: session_id, dwell_seconds (ignored here).
    """
    if recs.empty:
        return 0

    # Build rows to insert
    rows = []
    now = datetime.utcnow()  # keep consistent; DB can store in UTC
    for rid in recs["recipe_id"].astype(int).tolist():
        rows.append({
            "user_id": user_id,
            "recipe_id": rid,
            "event_type": "impression",
            "created_at": now,
            "session_id": session_id
        })

    # We’ll try insert with session_id column; if your schema doesn't have it, remove it below.
    insert_sql = text("""
        INSERT INTO events (user_id, recipe_id, event_type, created_at, session_id)
        VALUES (:user_id, :recipe_id, :event_type, :created_at, :session_id);
    """)

    # If your events table DOES NOT have session_id, use this instead:
    # insert_sql = text("""
    #     INSERT INTO events (user_id, recipe_id, event_type, created_at)
    #     VALUES (:user_id, :recipe_id, :event_type, :created_at);
    # """)

    with engine.begin() as conn:
        conn.execute(insert_sql, rows)

    return len(rows)


def main(user_id: int = 1, top_k: int = 10, write_impressions: bool = True, session_id: str | None = None):
    engine = create_engine(DB_URL)

    # 1) seed
    seed_id = get_seed_id(engine, user_id)
    if seed_id is None:
        print("No clicks found for this user yet.")
        return

    # 2) recipes
    df = get_recipes_df(engine)

    # 3) recent impressions
    shown_set = get_recent_impressions(engine, user_id, limit=20)

    # 4) compute recs
    recs = compute_recommendations(df, seed_id, shown_set, top_k=top_k)

    # 5) print
    seed_title = df.loc[df["recipe_id"].astype(int) == seed_id, "title"].iloc[0]
    print(f"\nSeed (last clicked): {seed_id} — {seed_title}\n")
    if recs.empty:
        print("No recommendations available after filtering.")
    else:
        print(recs[["recipe_id", "title", "cuisine", "category", "main_ingredient",
                    "prep_time_min", "health_score", "similarity"]].to_string(index=False))

    # 6) write back as impressions
    if write_impressions and not recs.empty:
        inserted = log_impressions(engine, user_id, recs, session_id=session_id)
        print(f"\n✅ Logged {inserted} recommendations as impressions into events.")


if __name__ == "__main__":
    # Example:
    # python src/recommend.py
    # or:
    # python src/recommend.py (and set session_id inside main call if you want)
    main(user_id=1, top_k=10, write_impressions=True, session_id=None)
