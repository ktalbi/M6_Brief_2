# M6_Brief_2 : MNIST Human-in-the-Loop – MLOps Pipeline

## Vue d’ensemble

Ce projet met en place une **pipeline MLOps complète** autour du dataset MNIST, intégrant :

- une **API FastAPI** pour l’inférence et la collecte de feedback utilisateur,
- une **application Streamlit** pour l’interface utilisateur,
- un **flow Prefect** pour le monitoring et le ré-entrainement automatique du modèle,
- **MLflow** pour le suivi des expériences et des modèles,
- **Prometheus & Grafana** pour l’observabilité.

Le cœur "intelligent" du système est le **processus de re-entrainement automatique** orchestré par `flow.py`, décrit en détail ci-dessous.

---

## Architecture (logique)

```
Utilisateur
   ↓ (image)
Streamlit → FastAPI → Modèle CNN
                    ↓
                Prédiction
                    ↓
        (optionnel) Correction humaine
                    ↓
               Base de données
                    ↓
               Prefect Flow
                    ↓
           Retrain + MLflow
                    ↓
             Reload API
```

---

## Objectif du retrain

Le re-entrainement vise à :

- corriger les **erreurs récurrentes** du modèle,
- intégrer les **retours humains** (Human-in-the-Loop),
- améliorer progressivement les performances **sans oublier** les connaissances initiales.

Le modèle **n’est jamais ré-entrainé uniquement sur les nouvelles données** : celles-ci sont utilisées comme **signal correctif**, ajoutées aux données MNIST historiques.

---

## Processus de retrain (flow.py)

Le fichier `flow.py` définit un **flow Prefect planifié**, exécuté périodiquement (par défaut toutes les heures, pour le test, toutes les 300s, verifier docker-compose  : FAIL_THRESHOLD: "1", MIN_NEW_LABELS: "1",  SCHEDULE_INTERVAL_SECONDS: "300").

### Chargement des feedbacks (`load_feedback`)

- Lecture de la base SQLite (`predictions.db`)
- Sélection des prédictions **corrigées par un humain**
- Conversion des images PNG → format MNIST `(28×28×1)` via le **préprocessing partagé** (`app.ml.model`)

---

### Analyse des erreurs (`compute_failure_counts`)

- Identification des prédictions incorrectes
- Comptage des erreurs totales et par classe

---

### Décision de retrain (`should_retrain`)

Le retrain est déclenché si :
- un nombre minimum de nouveaux labels est atteint
- au moins une classe dépasse un seuil d’erreurs

---

### Optimisation et entrainement

- Préparation des données : MNIST + feedback
- Optimisation Optuna
- Entrainement final
- Logging MLflow

---

### Déploiement

- Sauvegarde du modèle versionné
- Mise à jour de `latest.keras`
- Reload à chaud de l’API

---

## Lancer le projet

### Fichier .env :
```bash
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
FASTAPI_PORT=8080
STREAMLIT_PORT=8501
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```


```bash
docker compose up -d --build
```
Accès :

API : http://localhost:8000/docs

Streamlit : http://localhost:8501

MLflow : http://localhost:5000

Prefect UI : http://localhost:4200

Grafana : http://localhost:3000

---


