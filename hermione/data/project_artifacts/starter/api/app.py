import os
import json
from pathlib import Path
import pandas as pd
from joblib import load
import logging
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field
import uvicorn
from enum import Enum

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


def predict_new(x):
    model = load(os.path.join(DATA_DIR, "output", "titanic_model_rf.pkl"))
    p = model.get_preprocessing()

    x = p.clean_data(x)
    x = p.categ_encoding(x)

    columns = model.get_columns()
    for col in columns:
        if col not in x.columns:
            x[col] = 0
    return model.predict_proba(x)[:, 1]


class PassengerClass(int, Enum):
    first_class = 1
    second_class = 2
    third_class = 3


class Sex(str, Enum):
    male = "male"
    female = "female"


class Passenger(BaseModel):
    p_class: PassengerClass = Field(description="Passenger Class")
    sex: Sex = Field(description="Sex")
    age: int = Field(ge=0, le=120, description="Age")


class PredictionResult(BaseModel):
    probability_of_survival: float = Field(
        ge=0, le=1, description="Probability of survival"
    )


@app.post("/invocations", response_model=PredictionResult)
def predict(passenger: Passenger) -> PredictionResult:
    data = pd.DataFrame(
        [{"Pclass": passenger.p_class, "Sex": passenger.sex, "Age": passenger.age}]
    )
    prediction = predict_new(data)[0]
    return PredictionResult(probability_of_survival=float(prediction))


class HealthCheckResult(BaseModel):
    success: bool


@app.get("/health", response_model=HealthCheckResult)
def health_check() -> HealthCheckResult:
    return HealthCheckResult(success=True)


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=5000, reload=True, debug=True)
