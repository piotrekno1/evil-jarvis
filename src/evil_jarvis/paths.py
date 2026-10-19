"""Where the project keeps models and settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
ENV_FILE = PROJECT_ROOT / ".env"
