-- =========================
-- Food Recommendation DB Schema (Click-based / Implicit Feedback)
-- =========================

-- Clean re-run
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS recipe_ingredients;
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
  recipe_id        BIGINT PRIMARY KEY,   -- CSV id
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
-- For now: only main_ingredient => stored here as is_main=true
-- -------------------------
CREATE TABLE recipe_ingredients (
  recipe_id      BIGINT NOT NULL REFERENCES recipes(recipe_id) ON DELETE CASCADE,
  ingredient_id  BIGINT NOT NULL REFERENCES ingredients(ingredient_id) ON DELETE RESTRICT,
  is_main        BOOLEAN NOT NULL DEFAULT FALSE,
  PRIMARY KEY (recipe_id, ingredient_id)
);

-- -------------------------
-- 5) events (implicit feedback)
-- event_type: impression, click
-- dwell_seconds: optional (time on recipe detail page)
-- session_id: optional (ties events together in a browsing session)
-- -------------------------
CREATE TABLE events (
  event_id       BIGSERIAL PRIMARY KEY,
  user_id        BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  recipe_id      BIGINT NOT NULL REFERENCES recipes(recipe_id) ON DELETE CASCADE,
  event_type     TEXT NOT NULL CHECK (event_type IN ('impression', 'click')),
  dwell_seconds  INT CHECK (dwell_seconds IS NULL OR dwell_seconds >= 0),
  session_id     TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------
-- Indexes
-- -------------------------
CREATE INDEX idx_events_user_time ON events(user_id, created_at DESC);
CREATE INDEX idx_events_user_type_time ON events(user_id, event_type, created_at DESC);
CREATE INDEX idx_events_recipe_time ON events(recipe_id, created_at DESC);
CREATE INDEX idx_recipe_ingredients_recipe ON recipe_ingredients(recipe_id);
CREATE INDEX idx_ingredients_name ON ingredients(name);
