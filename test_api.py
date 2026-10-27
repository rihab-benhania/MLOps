import requests

# ===== Configuration =====
BASE_URL = "http://localhost:8000"

# ===== 1. Vérification de l'état de l'API (Health Check) =====
response = requests.get(f"{BASE_URL}/health")
print("GET /health ->", response.json())

# ===== 2. Test de prédiction : Client à risque =====
# Faible consommation et faible ancienneté
client_1 = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10,
}

response = requests.post(f"{BASE_URL}/predict", json=client_1)
print("POST /predict (client 1) ->", response.status_code, response.json())

# ===== 3. Test de prédiction : Client fidèle =====
# Forte consommation et forte ancienneté
client_2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80,
}

response = requests.post(f"{BASE_URL}/predict", json=client_2)
print("POST /predict (client 2) ->", response.status_code, response.json())

# ===== 4. Test de validation : Transmission de données invalides =====
# Copie du premier client en injectant un âge invalide (chaîne de caractères)
client_invalide = {**client_1, "age": "abc"}

response = requests.post(f"{BASE_URL}/predict", json=client_invalide)
print("POST /predict (age='abc') -> Code HTTP :", response.status_code)
print("Message d'erreur :", response.json()["detail"][0]["msg"])