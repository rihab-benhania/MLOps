from typing import Literal

from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

# ===== 1. Initialisation de l'application FastAPI =====
app = FastAPI(title="API Churn")

# ===== 2. Chargement du modèle « champion » au démarrage =====
# Le modèle est chargé une seule fois au démarrage pour optimiser les performances
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")


# ===== 3. Définition et validation du schéma d'entrée =====
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# ===== 4. Route /health : Vérification de l'état de l'API =====
@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 5. Route /predict : Traitement et prédiction du churn =====
@app.post("/predict")
def predict(client: Client):
    # a) Transformation des données reçues en DataFrame d'une seule ligne
    df_client = pd.DataFrame([client.model_dump()])

    # b) One-Hot Encoding de la variable catégorielle « abonnement »
    df_encoded = pd.get_dummies(df_client)

    # c) Alignment sur les colonnes exactes attendues par le modèle (valeurs manquantes à 0)
    df_prepared = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # d) Calcul de la prédiction (1 = départ, 0 = fidélité)
    prediction = int(model.predict(df_prepared)[0])

    return {
        "churn": prediction,
        "message": "Le client va partir" if prediction == 1 else "Le client reste",
    }