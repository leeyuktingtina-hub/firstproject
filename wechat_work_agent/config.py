import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = os.getenv("MODEL", "claude-sonnet-4-6")
MAX_SUBAGENTS = int(os.getenv("MAX_SUBAGENTS", "10"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
