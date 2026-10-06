"""Paths and analysis settings, read once from config/config.yaml (or the file named by $COWEAR_CONFIG).

Relative paths in the config resolve against the repository root, so the demo runs from a fresh clone.
"""

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = Path(os.environ.get("COWEAR_CONFIG", PROJECT_ROOT / "config" / "config.yaml"))

with open(CONFIG_FILE, "r", encoding="utf-8") as _f:
    _config = yaml.safe_load(_f)

DATA_ROOT = PROJECT_ROOT / _config["data_root"]
PREPROCESSED_ROOT = PROJECT_ROOT / _config["preprocessed_root"]
RBD_ROOT = PROJECT_ROOT / _config["sleep_root"]
AX6APPLE = _config["ax6apple"]
