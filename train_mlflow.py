# ===== 1. Importer les bibliothèques =====
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ===== 2. Charger les données =====
df = pd.read_csv("churn_data.csv")
y = df["churn"]


# ===== 3. Fonction : préparer les données =====
def preparer(colonnes):
    # Transforme les colonnes catégorielles (ex: 'abonnement') en variables muettes (0/1)
    X = pd.get_dummies(df[colonnes], drop_first=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ===== 4. Fonction : lancer UN essai et l'enregistrer dans MLflow =====
def lancer_essai(nom, n_estimators, max_depth, colonnes):
    X_train, X_test, y_train, y_test = preparer(colonnes)

    with mlflow.start_run(run_name=nom):
        # Entraînement du modèle
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # Calcul des métriques
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        # --- Paramètres MLflow ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))

        # --- Métriques MLflow ---
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        # --- Artefact (Modèle) MLflow ---
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

        print(f"{nom} -> accuracy : {round(acc, 3)} | f1 : {round(f1, 3)}")


# ===== 5. Programme principal =====
if __name__ == "__main__":
    mlflow.set_experiment("churn-prediction")

    TOUT = ["age", "salaire", "abonnement", "anciennete", "consommation"]

    # Essai 1 : 100 arbres, profondeur 5, toutes les colonnes
    lancer_essai("essai-1", 100, 5, TOUT)

    # Essai 2 : modèle plus simple (50 arbres, profondeur 3)
    lancer_essai("essai-2", 50, 3, TOUT)

    # Essai 3 : sans la colonne salaire
    lancer_essai(
        "essai-3",
        100,
        5,
        ["age", "abonnement", "anciennete", "consommation"],
    )