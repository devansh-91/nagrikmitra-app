"""
Central config: paths, thresholds, env loading.

Responsibilities:
- Load .env (python-dotenv) without ever requiring a key to run.
- Expose constants: CONFIDENCE_ABSTAIN_THRESHOLD=0.45, DOMAIN_MARGIN_THRESHOLD=0.08
- Resolve paths: DATA_DIR, MODELS_DIR, DB_PATH (data/nagrikmitra.db), TAXONOMY_YAML_PATH
- Read GROQ_API_KEY / GROQ_MODEL / OLLAMA_HOST / OLLAMA_MODEL, all optional.
- Read DATABASE_URL (optional): a Postgres connection string. If unset,
  nagrikmitra/store.py uses a local SQLite file instead — no setup needed
  for local dev/tests. Set it in production for data that survives redeploys.
- Read SESSION_SECRET / OFFICER_SIGNUP_CODE / GOOGLE_CLIENT_ID /
  GOOGLE_CLIENT_SECRET / GOOGLE_REDIRECT_URI for auth (nagrikmitra/auth.py),
  all optional except OFFICER_SIGNUP_CODE (officer registration is rejected
  if unset — see auth_routes.py).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Loads .env if present; silently no-ops otherwise (never required to run).
load_dotenv(BASE_DIR / ".env")

# --- Thresholds (mirror config/taxonomy.yaml::routing) ---
CONFIDENCE_ABSTAIN_THRESHOLD = 0.45
DOMAIN_MARGIN_THRESHOLD = 0.08

# --- Paths ---
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = DATA_DIR / "nagrikmitra.db"

# Optional Postgres connection string (e.g. a free Neon/Supabase database).
# If unset, store.py falls back to the local SQLite file above.
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
TAXONOMY_YAML_PATH = BASE_DIR / "config" / "taxonomy.yaml"
COMPLAINTS_CSV_PATH = DATA_DIR / "complaints.csv"
DOMAIN_MODEL_PATH = MODELS_DIR / "domain.joblib"
PRIORITY_MODEL_PATH = MODELS_DIR / "priority.joblib"
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"

# --- Optional LLM env ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b").strip()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2").strip()

# --- API ---
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# Render sets RENDER=true on every deployed service (always HTTPS there);
# unset locally, where dev runs over plain http://.
IS_RENDER = os.environ.get("RENDER", "").strip().lower() == "true"

# --- Auth ---
# SESSION_SECRET: if unset, nagrikmitra/auth.py generates one and persists it
# to data/.session_secret (gitignored) on first run, so sessions survive
# restarts without forcing setup. Set this explicitly for anything beyond
# local dev/demo use.
SESSION_SECRET = os.environ.get("SESSION_SECRET", "").strip()
SESSION_SECRET_FILE = DATA_DIR / ".session_secret"

# Officer self-registration requires this shared code; registering as an
# officer is rejected outright if it's unset (never silently open).
OFFICER_SIGNUP_CODE = os.environ.get("OFFICER_SIGNUP_CODE", "").strip()

# Optional "Sign in with Google" (nagrikmitra/auth.py). If either is unset,
# Google login is simply not offered — email/username+password still works
# fully without it. Create credentials at
# https://console.cloud.google.com/apis/credentials.
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.environ.get(
    "GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback"
).strip()
