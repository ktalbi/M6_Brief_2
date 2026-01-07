# MNIST Human-in-the-Loop (Streamlit + FastAPI + Prefect + Optuna + MLflow + Prometheus/Grafana)

Ce projet met en place une chaîne complète de **test en situation réelle** d'un modèle de classification MNIST avec **boucle de feedback humain** et **réentraînement périodique**.

## Architecture

- **Streamlit** : interface de dessin (canvas) + correction utilisateur.
- **FastAPI** : endpoint `/predict` (upload PNG 28×28) + `/correct` (correction) + stockage SQLite.
- **SQLite** (volume Docker) : stocke les images (PNG) + labels prédits/corrigés.
- **Prefect** : exécute toutes les heures une analyse des nouvelles données + déclenche un retrain si conditions.
- **Optuna** : optimisation d'hyperparamètres (lr, dropout, batch size).
- **MLflow** : tracking des runs (monitoring + retrain), versionnage des paramètres/métriques/artifacts.
- **Prometheus + Grafana** : métriques API (latence, statut HTTP) + métriques custom (predictions/corrections).

## Lancer le projet

```bash
docker compose up --build
```

- Streamlit : http://localhost:8501
- API FastAPI : http://localhost:8000 (docs: `/docs`)
- Prometheus : http://localhost:9090
- Grafana : http://localhost:3000 (login admin/admin par défaut)
- MLflow : http://localhost:5000

## Boucle de feedback

1. Dessiner un chiffre dans Streamlit.
2. Cliquer **Prédire** → l'API renvoie `{prediction_id, predicted_label, probabilities}` et stocke l'image + prédiction.
3. Si c'est faux, sélectionner le label correct et cliquer **Corriger & enregistrer**.
4. Toutes les heures (service `prefect`), le flow :
   - charge les corrections,
   - calcule les erreurs par classe,
   - calcule un score simple de drift,
   - déclenche un retrain si :
     - assez de nouvelles données (MIN_NEW_LABELS), ET
     - drift > DRIFT_THRESHOLD **ou** erreurs par classe ≥ FAIL_THRESHOLD.

## Versionning des modèles

- Les modèles sont écrits dans le volume `shared_models` :
  - `/app/models/model_<timestamp>.keras`
  - `/app/models/latest.keras` (copie du dernier modèle déployé)
- FastAPI recharge le modèle via `/reload`.

## Notes importantes

- Le drift ici est un **heuristique** (distance sur la distribution de l'intensité moyenne).  
  En production, vous remplacerez cela par Evidently/Alibi-Detect/KS tests multi-features, embeddings, etc.
- Pour accélérer, l'Optuna objective n'entraîne qu'**1 epoch** par trial.
