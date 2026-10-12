# Text Mining – TP2 : analyse textuelle multi-tâches, mono-langue et multilingue

Système d'analyse d'avis clients appliqué à deux jeux de données (anglais, puis anglais / français / darija), avec suivi des expériences via MLflow. L'énoncé est dans [`docs/enonce_tp2.pdf`](docs/enonce_tp2.pdf).

## Contenu

| Notebook | Dataset | Contenu |
|---|---|---|
| `notebooks/01_dataset1_mono_langue.ipynb` | `data/dataset1_reviews.csv` (500 avis, EN) | EDA, 2 pipelines de prétraitement, BoW / TF-IDF / Word2Vec / SBERT, LR et SVM, topic modeling (LDA, BERTopic), t-SNE, recommandation |
| `notebooks/02_dataset2_multilingue.ipynb` | `data/dataset2_multilingual_reviews.csv` (1000 avis, EN/FR/Darija) | mêmes étapes en contexte multilingue + analyse métier + comparaison DS1 vs DS2 |

Les graphiques générés sont dans `figures/dataset1/` et `figures/dataset2/`.

## Structure

```
text-mining-tp2/
├── notebooks/   # 01 (mono-langue), 02 (multilingue)
├── data/        # jeux de données CSV
├── figures/     # graphiques produits par les notebooks
├── docs/        # énoncé du TP
├── mlflow/      # base MLflow (créée à l'exécution, non versionnée)
├── requirements.txt
└── README.md
```

## Installation et exécution

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Lancer Jupyter depuis `notebooks/` (les chemins sont relatifs à ce dossier). Les modèles SBERT et les ressources NLTK sont téléchargés automatiquement au premier lancement.

Interface MLflow, depuis la racine du projet :

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db       # dataset 1
python -m mlflow ui --backend-store-uri sqlite:///mlflow/mlflow_ds2.db   # dataset 2
```

## Remarque sur les résultats

Chaque dataset ne contient que 15 textes distincts répétés (avec des variations de note et de catégorie). Les scores F1 de 1.0 obtenus par plusieurs modèles reflètent ce faible nombre d'exemples uniques, pas une généralisation réelle.
