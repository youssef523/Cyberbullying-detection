# Détection de harcèlement social (Cyberbullying Detection)

Projet de bout en bout : exploration des données, stockage MongoDB, analyse,
modélisation, comparaison de modèles, visualisation et déploiement via API
(FastAPI + Docker).

## Structure du projet

```
cyberbullying-detection/
│
├── data/
│   ├── raw/                          # Dataset brut téléchargé depuis Kaggle
│   └── processed/                    # Données nettoyées
│
├── notebooks/
│   ├── 01_exploration_nettoyage.ipynb
│   ├── 02_chargement_mongodb.ipynb
│   ├── 03_analyse_descriptive.ipynb
│   ├── 04_modelisation.ipynb
│   ├── 05_comparaison_modeles.ipynb
│   └── 06_visualisation_resultats.ipynb
│
├── models_artifacts/                 # Modèles entraînés (versionnés, prêts à l'emploi)
│   ├── tfidf_vectorizer.joblib
│   ├── svm_model.joblib
│   ├── logistic_regression_model.joblib
│   └── label_map.joblib
│
├── api/
│   ├── main.py                       # Point d'entrée FastAPI
│   ├── preprocessing.py              # Nettoyage de texte (identique au notebook 01)
│   └── schemas.py                    # Modèles Pydantic
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── requirements.txt
├── .env.example
└── .gitignore
```

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Dataset

Télécharger le dataset depuis Kaggle et le placer dans `data/raw/` :
https://www.kaggle.com/datasets/saifulislam7/cyberbullying-and-harassment-detection-using-ml

## Étapes du projet

1. **Exploration et nettoyage** — `notebooks/01_exploration_nettoyage.ipynb`
2. **Chargement dans MongoDB** — `notebooks/02_chargement_mongodb.ipynb`
3. **Analyse descriptive** — `notebooks/03_analyse_descriptive.ipynb`
4. **Modélisation** — `notebooks/04_modelisation.ipynb`
5. **Comparaison des modèles** — `notebooks/05_comparaison_modeles.ipynb`
6. **Visualisation des résultats** — `notebooks/06_visualisation_resultats.ipynb`
7. **Conteneurisation** — `api/` + `docker/`

## Modèles entraînés

Les modèles entraînés sont inclus dans `models_artifacts/` : l'API fonctionne
directement après un clone, sans réentraînement.

Pour les régénérer, placer le dataset dans `data/raw/` puis exécuter dans l'ordre
`notebooks/01_exploration_nettoyage.ipynb` et `notebooks/04_modelisation.ipynb`.

## Lancer l'API en local

```bash
uvicorn api.main:app --reload
```

Documentation interactive : http://localhost:8000/docs

### Endpoints

| Méthode | Route      | Description                                   |
|---------|------------|-----------------------------------------------|
| GET     | `/`        | Statut de l'API                               |
| GET     | `/health`  | Vérifie que les modèles sont chargés          |
| GET     | `/models`  | Liste les modèles disponibles                 |
| POST    | `/predict` | Prédit si un message relève du harcèlement    |

Exemple :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "you are a disgusting idiot", "model": "svm"}'
```

```json
{"text": "you are a disgusting idiot", "is_harassment": true, "confidence": 0.8977, "model_used": "svm"}
```

`model` accepte `svm` (par défaut) ou `logistic_regression`.

## Lancer avec Docker

```bash
cd docker
docker-compose up --build
```

L'API sera disponible sur http://localhost:8000

## Livrables

- [x] Code source complet du projet
- [x] Dockerfile
- [x] Documentation (ce README)
- [x] Étude comparative des deux modèles (`notebooks/05_comparaison_modeles.ipynb`)
