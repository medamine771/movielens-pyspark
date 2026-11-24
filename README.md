# Recommandation de films avec PySpark

Ce projet propose jusqu’à 10 films à chaque utilisateur de MovieLens à partir de ses notes. Il compare le filtrage collaboratif **ALS** de Spark MLlib à une référence qui classe les films selon le nombre de notes reçues.

## Lancer le projet

Il faut Python 3.10 ou plus récent et un JDK 17 ou 21. Sous Windows, ouvre PowerShell dans le dossier qui contient `recommend.py`. Vérifie que `JAVA_HOME` pointe vers le dossier du JDK et que `java -version` affiche la version attendue. [Installation de PySpark](https://spark.apache.org/docs/4.0.1/api/python/getting_started/install.html)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

L’archive du projet contient les deux fichiers CSV de MovieLens nécessaires et le `README.txt` de GroupLens dans `data/ml-latest-small/`. Tu peux aussi récupérer ces fichiers depuis l’archive officielle. [Téléchargement MovieLens](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) · [Conditions d’utilisation](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)

```powershell
.\.venv\Scripts\python.exe recommend.py --user 1
```

Sous macOS ou Linux, utilise `python3` pour créer l’environnement et `.venv/bin/python` pour les deux commandes suivantes.

`--user 1` choisit l’utilisateur affiché dans le terminal. Le CSV contient les recommandations de tous les utilisateurs. `--top-k 5` change la taille du classement et `--data-dir` permet de pointer vers un autre dossier contenant les deux CSV.

Si Windows signale une erreur liée à `TEMP`, définis `$env:TEMP` dans ton terminal sur le chemin d’un dossier temporaire existant et accessible en écriture, dont le chemin ne contient aucun espace, puis relance la commande.

## Lire les résultats

| Fichier | Contenu |
|---|---|
| `results/recommendations.csv` | Identifiant utilisateur, rang, identifiant du film, titre, genres et score ALS. Le score sert au classement et peut dépasser 5. |
| `results/metrics.json` | Résultats d’ALS et du classement par popularité, avec les effectifs du jeu d’entraînement et du jeu de test. |

- **Précision@K** mesure la proportion des K places occupées par des films pertinents.
- **Rappel@K** mesure la proportion des films pertinents retrouvés parmi les K recommandations.
- **Hit rate@K** mesure la proportion des utilisateurs évalués qui retrouvent au moins un film pertinent.

L’exécution fournie a produit 6 100 recommandations pour 610 utilisateurs. Parmi les 591 utilisateurs évalués, le classement par popularité retrouve au moins un film pertinent pour 31,13 % d’entre eux, contre 6,43 % pour ALS avec les paramètres du script.

## Comprendre le pipeline

1. **Lire les données.** Spark charge les notes et le catalogue avec des schémas explicites.
2. **Préparer le test.** Pour chaque utilisateur, les 20 % de notes les plus récentes, arrondis à l’entier inférieur avec un minimum d’une note, servent au test. Un utilisateur avec une seule note reste dans l’entraînement. Une note de 4/5 ou plus dans le test rend le film pertinent.
3. **Comparer les méthodes.** ALS apprend à représenter les utilisateurs et les films à partir des notes d’entraînement. Le classement de référence compte les notes d’entraînement par film. Les deux méthodes écartent les films déjà notés dans cet historique. Les métriques sont moyennées sur les utilisateurs ayant au moins un film pertinent dans le test. Les films du test absents de l’entraînement restent comptés parmi les films pertinents. [Documentation Spark sur ALS](https://spark.apache.org/docs/4.0.1/ml-collaborative-filtering.html)
4. **Générer les recommandations finales.** Après l’évaluation, le script réentraîne ALS sur toutes les notes, retire les films déjà notés et ajoute les titres et les genres avant l’export.

Le découpage chronologique est fait par utilisateur. Les dates de coupure peuvent donc différer d’un utilisateur à l’autre.

Les données proviennent de GroupLens. Référence à citer si tu publies des résultats obtenus avec ce jeu de données. [Harper et Konstan, *The MovieLens Datasets: History and Context*, 2015](https://doi.org/10.1145/2827872)
