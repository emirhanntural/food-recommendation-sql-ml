-- =========================
-- Food Recommendation DB Schema (PostgreSQL)
-- =========================

-- Clean re-run
DROP TABLE IF EXISTS recipe_ingredients;
DROP TABLE IF EXISTS feedback;
DROP TABLE IF EXISTS ingredients;
DROP TABLE IF EXISTS recipes;
DROP TABLE IF EXISTS users;

-- -------------------------
-- 1) users
-- -------------------------
CREATE TABLE users (
  user_id      BIGSERIAL PRIMARY KEY,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------
-- 2) recipes
-- -------------------------
CREATE TABLE recipes (
  recipe_id        BIGINT PRIMARY KEY,   -- we will use CSV id as recipe_id
  title            TEXT NOT NULL,
  prep_time_min    INT,
  category         TEXT,
  cuisine          TEXT,
  health_score     INT,
  difficulty       TEXT,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------
-- 3) ingredients
-- -------------------------
CREATE TABLE ingredients (
  ingredient_id  BIGSERIAL PRIMARY KEY,
  name           TEXT NOT NULL UNIQUE
);

-- -------------------------
-- 4) recipe_ingredients (many-to-many)
-- For now we'll store only main_ingredient from the CSV.
-- Later you can expand to multiple ingredients per recipe.
-- -------------------------
CREATE TABLE recipe_ingredients (
  recipe_id      BIGINT NOT NULL REFERENCES recipes(recipe_id) ON DELETE CASCADE,
  ingredient_id  BIGINT NOT NULL REFERENCES ingredients(ingredient_id) ON DELETE RESTRICT,
  is_main        BOOLEAN NOT NULL DEFAULT FALSE,
  PRIMARY KEY (recipe_id, ingredient_id)
);

-- -------------------------
-- 5) feedback (labels for ML)
-- label: 1=like, 0=dislike
-- -------------------------
CREATE TABLE feedback (
  feedback_id   BIGSERIAL PRIMARY KEY,
  user_id       BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  recipe_id     BIGINT NOT NULL REFERENCES recipes(recipe_id) ON DELETE CASCADE,
  label         SMALLINT NOT NULL CHECK (label IN (0, 1)),
  source        TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------
-- Indexes (performance)
-- -------------------------
CREATE INDEX idx_feedback_user_time ON feedback(user_id, created_at DESC);
CREATE INDEX idx_feedback_user_recipe ON feedback(user_id, recipe_id);
CREATE INDEX idx_recipe_ingredients_recipe ON recipe_ingredients(recipe_id);
CREATE INDEX idx_ingredients_name ON ingredients(name);

