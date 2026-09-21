"""
Central config: paths, thresholds, env loading.

Responsibilities:
- Load .env (python-dotenv) without ever requiring a key to run.
- Expose constants: CONFIDENCE_ABSTAIN_THRESHOLD=0.45, DOMAIN_MARGIN_THRESHOLD=0.08
- Resolve paths: DATA_DIR, MODELS_DIR, DB_PATH (data/nagrikmitra.db), TAXONOMY_YAML_PATH
- Read GROQ_API_KEY / GROQ_MODEL / OLLAMA_HOST / OLLAMA_MODEL, all optional.
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
