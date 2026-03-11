"""pytest configuration — adds src/ to Python path."""
import sys
import os

# Allow imports like `from src.config import ...`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
