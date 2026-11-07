"""
main.py - Главная точка входа в систему Lex.uz Legal RAG.
"""

import os
import sys
from pathlib import Path

# Enable expandable segments for PyTorch CUDA allocator to prevent VRAM fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    _env_file = Path(__file__).resolve().parent / ".env"
    if _env_file.exists():
        with open(_env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

# Auto-bootstrap project virtual environment if invoked via system Python
_venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
if _venv_python.exists() and sys.prefix != str(Path(__file__).resolve().parent / ".venv"):
    os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

from lexuz.cli import main

if __name__ == "__main__":
    main()


