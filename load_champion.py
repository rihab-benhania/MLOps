import mlflow
import mlflow.sklearn  # Pour recharger un modèle scikit-learn
import pandas as pd

# ===== 1. Configuration de la base de suivi =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 2. Charger le modèle « champion » par son alias =====
# Chargement sans dépendre directement d'un numéro de version ou d'un ID de run
model_uri = "models:/churn-model@champion"
model = mlflow.sklearn.load_model(model_uri)

print("Modèle chargé :", type(model).__name__)
print("Variables attendues :", list(model.feature_names_in_))

# ===== 3. Préparer les données d'un nouveau client =====
# Données brutes de test (les colonnes d'abonnement sont déjà encodées en booléen/binaires)
client_data = {
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True,
}

client = pd.DataFrame([client_data])

# Réordonner et filtrer les colonnes pour correspondre exactement aux attentes du modèle
client = client[list(model.feature_names_in_)]

# ===== 4. Prédiction =====
# 1 = Le client va partir (churn), 0 = Le client reste
prediction = model.predict(client)[0]
resultat = "OUI" if prediction == 1 else "NON"

print(f"Le client va-t-il partir ? -> {resultat}")