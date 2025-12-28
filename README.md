# Click-Based Food Recommendation System

This project is a small, end-to-end click-based (implicit feedback) food recommendation prototype.

Instead of ratings or likes, the system learns from user behavior:
- which recipes are shown (impression)
- which recipes are clicked (click)

Recommendations are generated based on the user’s most recent behavior and logged back to the system.

---

## How it works

1. Recipes are shown to the user → impression
2. User clicks a recipe → click
3. The system:
   - takes the most recent click (or weighted recent clicks),
   - finds similar recipes using content-based similarity,
   - recommends top results,
   - logs them back as new impression events

This creates a simple recommendation feedback loop.

---

## Recommendation logic

- Type: Content-based
- Seed: Last click (optionally weighted recent clicks)
- Features:
  - main ingredient
  - cuisine
  - category
  - difficulty
  - preparation time (normalized)
  - health score (normalized)
- Similarity: Cosine similarity (one-hot + numeric features)
- Filters:
  - excludes recently shown recipes
  - excludes already clicked items

---

## Tech stack

- Python
- PostgreSQL
- Pandas
- scikit-learn
- Streamlit

---

## Project structure


---

## How to run (local)

1. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

export DATABASE_URL="postgresql://localhost:5432/foodrec"

streamlit run app.py
