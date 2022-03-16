import os
import json
from pathlib import Path
import logging
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field
import uvicorn

logging.getLogger().setLevel(logging.INFO)

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT_DIR, "data"))

with open(os.path.join(PROJECT_ROOT_DIR, "config", "config.json"), "r") as file:
    SETTINGS = json.load(file)

app = FastAPI(
    title=SETTINGS["project_name"],
    redoc_url=SETTINGS["docs_url"] if SETTINGS["use_redocs"] else None,
    docs_url=SETTINGS["docs_url"] if not SETTINGS["use_redocs"] else None,
)


class HealthCheckResult(BaseModel):
    success: bool


@app.get("/health", response_model=HealthCheckResult)
def health_check() -> HealthCheckResult:
    return HealthCheckResult(success=True)


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=5000, reload=True, debug=True)
