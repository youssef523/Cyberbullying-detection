"""
Modèles Pydantic pour les requêtes/réponses de l'API.
"""

from typing import Literal

from pydantic import BaseModel, Field

ModelName = Literal["svm", "logistic_regression"]


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Message à analyser")
    model: ModelName = Field("svm", description="Modèle à utiliser pour la prédiction")


class PredictionResponse(BaseModel):
    text: str
    is_harassment: bool
    confidence: float
    model_used: str


class ModelsResponse(BaseModel):
    available_models: list[ModelName]
    default_model: ModelName
