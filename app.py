import os
import uuid
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/foodrec")

# ---------- Helpers ----------
def get_engine():
    return create_engine(DB_URL)

def ensure_user(engine, user_id: int | None) -> int:
    if user_id is not None:
        return int(user_id)
    with engine.begin() as conn:
        uid = conn.execute(text("INSERT INTO users DEFAULT VALUES RETURNING user_id;")).scalar_one()
    return int(uid)

def fetch_feed(engine, n=10, exclude_recent=20, user_id=1) -> pd.DataFrame:
    # Son gösterilenleri tekrar göstermeyelim
    shown = pd.read_sql(
        text("""
        SELECT recipe_id
        FROM events
        WHERE user_id=:uid AND event_type='impression'
        ORDER BY created_at DESC
        LIMIT :lim
        """),
        engine,
        params={"uid": user_id, "lim": exclude_recent}
    )
    shown_set = set(shown["recipe_id"].astype(int).tolist()) if not shown.empty else set()

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

    if shown_set:
        df = df[~df["recipe_id"].astype(int).isin(shown_set)]

    # rastgele feed
    if len(df) > n:
        df = df.sample(n=n, random_state=None)

    return df.reset_index(drop=True)

def log_impressions(engine, user_id: int, recipe_ids: list[int], session_id: str):
    if not recipe_ids:
        return
    rows = [{"user_id": user_id, "recipe_id": int(rid), "session_id": session_id} for rid in recipe_ids]
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO events (user_id, recipe_id, event_type, session_id)
            VALUES (:user_id, :recipe_id, 'impression', :session_id);
            """),
            rows
        )

def log_click(engine, user_id: int, recipe_id: int, session_id: str, dwell_seconds: int | None = None):
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO events (user_id, recipe_id, event_type, dwell_seconds, session_id)
            VALUES (:user_id, :recipe_id, 'click', :dwell_seconds, :session_id);
            """),
            {"user_id": user_id, "recipe_id": int(recipe_id), "dwell_seconds": dwell_seconds, "session_id": session_id}
        )

def get_last_clicked(engine, user_id: int) -> int | None:
    res = pd.read_sql(
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
    if res.empty:
        return None
    return int(res.loc[0, "recipe_id"])

def make_feature_text(df: pd.DataFrame) -> pd.Series:
    def clean(x):
        if x is None:
            return ""
        return str(x).strip().lower().replace(" ", "_")

    return (
        "main_" + df["main_ingredient"].fillna("").map(clean) + " "
        "cuisine_" + df["cuisine"].fillna("").map(clean) + " "
        "cat_" + df["category"].fillna("").map(clean) + " "
        "diff_" + df["difficulty"].fillna("").map(clean)
    )

def recommend_for_seed(engine, seed_id: int, user_id: int, top_k=10, exclude_recent_impressions=20) -> pd.DataFrame:
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

    # son impression'ları çıkar
    shown = pd.read_sql(
        text("""
        SELECT recipe_id
        FROM events
        WHERE user_id=:uid AND event_type='impression'
        ORDER BY created_at DESC
        LIMIT :lim;
        """),
        engine,
        params={"uid": user_id, "lim": exclude_recent_impressions}
    )
    shown_set = set(shown["recipe_id"].astype(int).tolist()) if not shown.empty else set()

    # numeric clean
    df["prep_time_min"] = pd.to_numeric(df["prep_time_min"], errors="coerce")
    df["health_score"] = pd.to_numeric(df["health_score"], errors="coerce")
    df["prep_time_min"] = df["prep_time_min"].fillna(df["prep_time_min"].median())
    df["health_score"] = df["health_score"].fillna(df["health_score"].median())

    feat_text = make_feature_text(df)
    X_cat = feat_text.str.get_dummies(sep=" ")

    def z(x):
        x = x.astype(float)
        return (x - x.mean()) / (x.std() + 1e-9)

    X_cat["z_prep_time"] = z(df["prep_time_min"])
    X_cat["z_health"] = z(df["health_score"])

    X = X_cat.values
    idx_map = {rid: i for i, rid in enumerate(df["recipe_id"].astype(int).tolist())}

    if seed_id not in idx_map:
        return pd.DataFrame()

    seed_idx = idx_map[seed_id]
    sims = cosine_similarity(X[seed_idx:seed_idx+1], X).ravel()
    df["similarity"] = sims

    recs = df[df["recipe_id"].astype(int) != seed_id].copy()
    if shown_set:
        recs = recs[~recs["recipe_id"].astype(int).isin(shown_set)]
    recs = recs.sort_values(["similarity", "health_score"], ascending=[False, False]).head(top_k)
    return recs.reset_index(drop=True)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Food Recommender (Click-based)", layout="wide")
st.title("🍽️ Food Recommender — Click-based Demo (Streamlit + Postgres)")

engine = get_engine()

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "user_id" not in st.session_state:
    st.session_state.user_id = ensure_user(engine, None)

if "feed_df" not in st.session_state:
    st.session_state.feed_df = pd.DataFrame()

if "seed_id" not in st.session_state:
    st.session_state.seed_id = get_last_clicked(engine, st.session_state.user_id)

with st.sidebar:
    st.subheader("Settings")
    st.write(f"Session: `{st.session_state.session_id}`")
    st.write(f"User ID: `{st.session_state.user_id}`")
    feed_n = st.slider("Feed size", 5, 20, 10)
    top_k = st.slider("Top-K recommendations", 5, 20, 10)
    if st.button("New session"):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.feed_df = pd.DataFrame()
        st.session_state.seed_id = get_last_clicked(engine, st.session_state.user_id)

    if st.button("Create new user"):
        st.session_state.user_id = ensure_user(engine, None)
        st.session_state.feed_df = pd.DataFrame()
        st.session_state.seed_id = None

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Feed (random)")
    if st.button("Generate feed"):
        feed = fetch_feed(engine, n=feed_n, exclude_recent=20, user_id=st.session_state.user_id)
        st.session_state.feed_df = feed

        # log impressions
        log_impressions(
            engine,
            user_id=st.session_state.user_id,
            recipe_ids=feed["recipe_id"].astype(int).tolist(),
            session_id=st.session_state.session_id
        )

    if st.session_state.feed_df.empty:
        st.info("Feed oluşturmak için soldaki **Generate feed** butonuna bas.")
    else:
        for _, row in st.session_state.feed_df.iterrows():
            rid = int(row["recipe_id"])
            title = row["title"]
            meta = f"{row.get('cuisine','')} • {row.get('category','')} • {row.get('difficulty','')} • {row.get('prep_time_min','')} min"
            with st.container(border=True):
                st.markdown(f"### {title}")
                st.caption(meta)
                st.write(f"Main ingredient: **{row.get('main_ingredient','')}** | Health: **{row.get('health_score','')}**")
                if st.button(f" View / Click (recipe {rid})", key=f"click_{rid}"):
                    # log click
                    log_click(engine, st.session_state.user_id, rid, st.session_state.session_id, dwell_seconds=None)
                    st.session_state.seed_id = rid
                    st.success(f"Click logged. Seed set to recipe_id={rid}")

with col2:
    st.subheader(" Recommendations (based on last click)")
    if st.session_state.seed_id is None:
        st.warning("Henüz click yok. Feed’den bir yemeğe tıkla.")
    else:
        seed_id = st.session_state.seed_id
        seed_title = pd.read_sql(
            text("SELECT title FROM recipes WHERE recipe_id=:rid"),
            engine,
            params={"rid": seed_id}
        ).iloc[0, 0]
        st.write(f"Seed: **{seed_title}** (`{seed_id}`)")

        recs = recommend_for_seed(engine, seed_id=seed_id, user_id=st.session_state.user_id, top_k=top_k)
        if recs.empty:
            st.info("Öneri üretilemedi (seed bulunamadı veya veri eksik).")
        else:
            st.dataframe(
                recs[["recipe_id","title","cuisine","category","main_ingredient","prep_time_min","health_score","similarity"]],
                use_container_width=True,
                hide_index=True
            )

st.divider()
st.subheader("Quick Stats (this user)")
stats = pd.read_sql(
    text("""
    SELECT event_type, COUNT(*) AS cnt
    FROM events
    WHERE user_id=:uid
    GROUP BY event_type
    ORDER BY event_type;
    """),
    engine,
    params={"uid": st.session_state.user_id}
)
st.dataframe(stats, use_container_width=True, hide_index=True)
