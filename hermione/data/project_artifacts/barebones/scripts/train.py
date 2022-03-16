import json
import os
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT_DIR, "data"))

with open(os.path.join(PROJECT_ROOT_DIR, "config", "config.json"), "r") as file:
    project_name = json.load(file)["project_name"]

## Set your mlflow experiment and your train code HERE
