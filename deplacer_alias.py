import mlflow
from mlflow import MlflowClient

# ===== 1. Connexion à la base de suivi MLflow =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# ===== 2. Retrouver le run spécifique « essai-2 » =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'",
)
run_id = runs.iloc[0]["run_id"]

# ===== 3. Enregistrer une nouvelle version du modèle =====
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model", name="churn-model"
)
print(f"Nouvelle version créée : {model_version.version}")

# ===== 4. Consulter la version actuellement marquée comme « champion » =====
champion_actuel = client.get_model_version_by_alias(
    name="churn-model", alias="champion"
)
print(f"Champion actuel : version {champion_actuel.version}")

# ===== 5. Déplacer temporairement l'alias puis le réinitialiser =====
# Déplacement de l'alias vers la nouvelle version
client.set_registered_model_alias(
    name="churn-model", alias="champion", version=model_version.version
)
champion_temp = client.get_model_version_by_alias(
    name="churn-model", alias="champion"
)
print(f"Après déplacement : champion -> version {champion_temp.version}")

# Rétablissement de l'alias vers la version d'origine
client.set_registered_model_alias(
    name="churn-model", alias="champion", version=champion_actuel.version
)
champion_restaure = client.get_model_version_by_alias(
    name="churn-model", alias="champion"
)
print(f"Après retour : champion -> version {champion_restaure.version}")