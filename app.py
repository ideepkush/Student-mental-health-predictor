import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path

BASE_DIR = Path(__file__).parent

app = FastAPI(
    title="Mental Health Score Predictor",
    description="Predicts a student's mental health score based on social media usage and lifestyle factors.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(BASE_DIR / "models" / "knn_mental_health.pkl")
_model_loaded = True


class StudentFeatures(BaseModel):
    Age: int = Field(..., ge=18, le=24, example=21)
    Gender: Literal["Male", "Female"]
    Country: Literal[
        "India", "USA", "Canada", "Australia", "UK",
        "Germany", "Mexico", "Turkey", "France", "Other"
    ]
    Academic_Level: Literal["High School", "Undergraduate", "Graduate"]
    Most_Used_Platform: Literal[
        "Instagram", "TikTok", "Facebook", "LinkedIn",
        "YouTube", "Twitter", "Snapchat", "WhatsApp",
        "LINE", "VKontakte", "KakaoTalk", "WeChat"
    ]
    Purpose_Of_Use: Literal["Entertainment", "Education", "Networking", "News"]
    Avg_Daily_Usage_Hours: float = Field(..., ge=1.0, le=8.8, example=4.0)
    Study_Hours: float = Field(..., ge=0.3, le=8.3, example=3.0)
    Physical_Activity_Hours: float = Field(..., ge=0.0, le=4.1, example=1.5)
    Sleep_Hours_Per_Night: float = Field(..., ge=3.6, le=9.9, example=7.0)
    Stress_Level: Literal["Low", "Medium", "High", "Very High"]


class PredictionResponse(BaseModel):
    mental_health_score: float
    interpretation: str


def interpret_score(score: float) -> str:
    if score >= 8.0:
        return "Excellent mental health"
    elif score >= 6.5:
        return "Good mental health"
    elif score >= 5.0:
        return "Moderate mental health — consider reducing screen time or improving sleep"
    else:
        return "Poor mental health — consider seeking support or making lifestyle changes"


@app.get("/")
def root():
    return {
        "name": "Student Social Media and Mental Health Score Predictor",
        "version": "1.0.0",
        "description": "Predicts a student's mental health score based on social media usage and lifestyle factors.",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model_loaded}


@app.get("/ui", include_in_schema=False)
def serve_ui():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures):
    input_df = pd.DataFrame([features.model_dump()])

    score = round(float(model.predict(input_df)[0]), 3)
    score = float(np.clip(score, 3.6, 9.4))

    return PredictionResponse(
        mental_health_score=score,
        interpretation=interpret_score(score)
    )
