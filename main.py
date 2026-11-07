"""
main.py - Главная точка входа в систему Lex.uz Legal RAG.
"""

import os
import sys
from pathlib import Path

# Enable expandable segments for PyTorch CUDA allocator to prevent VRAM fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Auto-bootstrap project virtual environment if invoked via system Python
_venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
if _venv_python.exists() and sys.prefix != str(Path(__file__).resolve().parent / ".venv"):
    os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

from lexuz.cli import main

if __name__ == "__main__":
    main()


