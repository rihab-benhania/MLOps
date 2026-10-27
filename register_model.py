# ===== 1. Importer les bibliothèques =====
import mlflow
from mlflow import MlflowClient

# ===== 2. Se connecter à la base de données de suivi =====
# Fichier mlflow.db du dossier courant (contient l'historique des runs)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Retrouver le meilleur run =====
# Tri par score F1 décroissant pour obtenir le meilleur modèle en première position
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"], 
    order_by=["metrics.f1 DESC"]
)

best = runs.iloc[0]
run_id = best["run_id"]

print(f"Meilleur run : {best['tags.mlflow.runName']} | id : {run_id[:8]}...")

# ===== 4. Enregistrer le modèle dans le Model Registry =====
# "runs:/<id>/model" pointe vers les artefacts du modèle
model_uri = f"runs:/{run_id}/model"
model_name = "churn-model"

result = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Modèle enregistré : {model_name} | version {result.version}")

# ===== 5. Définir l'alias « champion » sur cette version =====
client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=result.version
)
print(f"Alias 'champion' -> version {result.version}")