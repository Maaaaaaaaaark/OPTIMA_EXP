"""Shared fixtures for the OPTIMA pipeline tests.

Adds the project root to sys.path so the tests can be run from anywhere:

    python -m pytest tests/ -v
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

GEMMA_CONFIG = os.path.join(CONFIG_DIR, "gemma2-2b", "hotpot_qa.yaml")
GEMMA_SMOKE_CONFIG = os.path.join(CONFIG_DIR, "gemma2-2b", "hotpot_qa_colab_smoke.yaml")
QWEN15_CONFIG = os.path.join(CONFIG_DIR, "qwen1.5b", "hotpot_qa.yaml")
QWEN05_CONFIG = os.path.join(CONFIG_DIR, "qwen0.5b", "hotpot_qa.yaml")
