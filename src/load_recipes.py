import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ---- Config ----
DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/foodrec")
CSV_PATH = os.getenv("RECIPES_CSV", "data/recipes.csv")

def normalize_ingredient(x: str) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()

def main():
    # 1) Read CSV
    df = pd.read_csv(CSV_PATH)

    # Basic sanity: expected columns
    expected = {"id", "title", "prep_time_min", "category", "cuisine", "health_score", "difficulty", "main_ingredient"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # 2) Connect DB
    engine = create_engine(DB_URL)

    # 3) Load recipes (upsert)
    recipes = df[["id","title","prep_time_min","category","cuisine","health_score","difficulty"]].copy()
    recipes = recipes.rename(columns={"id": "recipe_id"})

    upsert_recipes_sql = text("""
        INSERT INTO recipes (recipe_id, title, prep_time_min, category, cuisine, health_score, difficulty)
        VALUES (:recipe_id, :title, :prep_time_min, :category, :cuisine, :health_score, :difficulty)
        ON CONFLICT (recipe_id) DO UPDATE SET
          title = EXCLUDED.title,
          prep_time_min = EXCLUDED.prep_time_min,
          category = EXCLUDED.category,
          cuisine = EXCLUDED.cuisine,
          health_score = EXCLUDED.health_score,
          difficulty = EXCLUDED.difficulty;
    """)

    with engine.begin() as conn:
        conn.execute(text("SET TIME ZONE 'UTC';"))

        # Insert recipes
        conn.execute(upsert_recipes_sql, recipes.to_dict(orient="records"))

        # 4) Insert ingredients (unique)
        ing_names = (
            df["main_ingredient"]
            .map(normalize_ingredient)
            .replace("", pd.NA)
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        conn.execute(
            text("""
                INSERT INTO ingredients (name)
                VALUES (:name)
                ON CONFLICT (name) DO NOTHING;
            """),
            [{"name": n} for n in ing_names]
        )

        # 5) Link recipe -> ingredient (is_main=true)
        # We need ingredient_id for each main_ingredient
        ing_map = dict(
            conn.execute(text("SELECT ingredient_id, name FROM ingredients")).fetchall()
        )

        links = []
        for _, row in df.iterrows():
            recipe_id = int(row["id"])
            name = normalize_ingredient(row["main_ingredient"])
            if not name:
                continue
            ingredient_id = ing_map.get(name)
            if ingredient_id is None:
                continue
            links.append({"recipe_id": recipe_id, "ingredient_id": int(ingredient_id)})

        conn.execute(
            text("""
                INSERT INTO recipe_ingredients (recipe_id, ingredient_id, is_main)
                VALUES (:recipe_id, :ingredient_id, TRUE)
                ON CONFLICT (recipe_id, ingredient_id) DO UPDATE SET
                  is_main = TRUE;
            """),
            links
        )

    print("Load complete.")
    print(f"Recipes inserted/updated: {len(recipes)}")
    print(f"Unique main_ingredients inserted: {len(ing_names)}")
    print(f"Recipe-ingredient links: {len(links)}")
    print("DB:", DB_URL)
    print("CSV:", CSV_PATH)

if __name__ == "__main__":
    main()
