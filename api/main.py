"""
Point d'entrée de l'API FastAPI pour la détection de harcèlement social.

Étape 7 du projet : conteneurisation avec Docker et FastAPI.
Charge les artefacts entraînés et les expose via un endpoint de prédiction.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException

from api.preprocessing import clean_text
from api.schemas import ModelsResponse, PredictionRequest, PredictionResponse

ARTIFACTS_DIR = Path(os.getenv("MODELS_ARTIFACTS_DIR", "models_artifacts"))

ARTIFACT_FILES = {
    "vectorizer": "tfidf_vectorizer.joblib",
    "svm": "svm_model.joblib",
    "logistic_regression": "logistic_regression_model.joblib",
    "label_map": "label_map.joblib",
}

artifacts: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for key, filename in ARTIFACT_FILES.items():
        path = ARTIFACTS_DIR / filename
        if not path.exists():
            print(f"[warmup] Artefact manquant : {path} (exécutez le notebook 04 avant de lancer l'API)")
            continue
        artifacts[key] = joblib.load(path)

    if "label_map" in artifacts:
        artifacts["inv_label_map"] = {v: k for k, v in artifacts["label_map"].items()}

    yield
    artifacts.clear()


app = FastAPI(
    title="Cyberbullying Detection API",
    description="API de détection de harcèlement social à partir de messages texte.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "API de détection de harcèlement social"}


@app.get("/health")
def health_check():
    models_ready = "vectorizer" in artifacts and any(m in artifacts for m in ("svm", "logistic_regression"))
    return {"status": "healthy" if models_ready else "degraded", "models_loaded": models_ready}


@app.get("/models", response_model=ModelsResponse)
def list_models():
    available = [m for m in ("svm", "logistic_regression") if m in artifacts]
    if not available:
        raise HTTPException(status_code=503, detail="Aucun modèle chargé. Vérifiez models_artifacts/.")
    return ModelsResponse(available_models=available, default_model="svm")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if "vectorizer" not in artifacts or request.model not in artifacts:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Modèle '{request.model}' indisponible. "
                "Entraînez et sauvegardez les modèles via notebooks/04_modelisation.ipynb "
                "puis redémarrez l'API."
            ),
        )

    cleaned = clean_text(request.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Le texte est vide après nettoyage.")

    vectorizer = artifacts["vectorizer"]
    model = artifacts[request.model]

    X = vectorizer.transform([cleaned])
    prediction = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][prediction])

    return PredictionResponse(
        text=request.text,
        is_harassment=bool(prediction == 1),
        confidence=round(proba, 4),
        model_used=request.model,
    )
