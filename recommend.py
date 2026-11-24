"""Entraîne un moteur de recommandation de films et évalue son top K."""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, SparkSession, Window, functions as F


PROJECT_DIR = Path(__file__).resolve().parent
RELEVANT_RATING = 4.0
TEST_FRACTION = 0.2
SEED = 42


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("la valeur doit être supérieure ou égale à 1")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data/ml-latest-small")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--top-k", type=positive_int, default=10)
    parser.add_argument("--user", type=positive_int, default=1, help="utilisateur affiché en exemple")
    return parser.parse_args()


def configure_windows_temp() -> None:
    if os.name != "nt":
        return
    temp_dir = os.environ.get("TEMP") or tempfile.gettempdir()
    if " " in temp_dir:
        # Le lanceur .cmd de Spark 4.0.1 ne protège pas le chemin de son fichier temporaire.
        import ctypes

        buffer = ctypes.create_unicode_buffer(32768)
        length = ctypes.windll.kernel32.GetShortPathNameW(temp_dir, buffer, len(buffer))
        if not 0 < length < len(buffer) or " " in buffer.value:
            raise SystemExit("Spark sous Windows requiert ici un dossier TEMP sans espace. Voir README.md.")
        temp_dir = buffer.value
    os.environ["TEMP"] = temp_dir


def load_data(spark: SparkSession, directory: Path) -> tuple[DataFrame, DataFrame]:
    reader = spark.read.options(header=True, mode="FAILFAST", enforceSchema=False, escape='"')
    ratings = reader.schema("userId INT, movieId INT, rating DOUBLE, timestamp LONG").csv(
        (directory / "ratings.csv").resolve().as_posix()
    )
    movies = reader.schema("movieId INT, title STRING, genres STRING").csv(
        (directory / "movies.csv").resolve().as_posix()
    )
    return ratings, movies


def split_ratings(ratings: DataFrame) -> tuple[DataFrame, DataFrame]:
    by_user = Window.partitionBy("userId")
    newest_first = by_user.orderBy(F.desc("timestamp"), F.desc("movieId"))
    labelled = (ratings
        .withColumn("user_ratings", F.count("*").over(by_user))
        .withColumn("recent_rank", F.row_number().over(newest_first)))
    test_size = F.greatest(F.lit(1), F.floor(F.col("user_ratings") * TEST_FRACTION))
    in_test = (F.col("user_ratings") > 1) & (F.col("recent_rank") <= test_size)
    return labelled.where(~in_test).select(*ratings.columns), labelled.where(in_test).select(*ratings.columns)


def train_als(ratings: DataFrame) -> ALSModel:
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=16,
        maxIter=10,
        regParam=0.1,
        numUserBlocks=4,
        numItemBlocks=4,
        seed=SEED,
    )
    return als.fit(ratings)


def candidate_limit(history: DataFrame, k: int) -> int:
    largest_history = history.groupBy("userId").count().agg(F.max("count")).first()[0]
    return k + largest_history


def unseen_top_k(candidates: DataFrame, history: DataFrame, k: int) -> DataFrame:
    ranking = Window.partitionBy("userId").orderBy(F.desc("score"), F.asc("movieId"))
    return (candidates
        .join(history.select("userId", "movieId"), ["userId", "movieId"], "left_anti")
        .withColumn("rank", F.row_number().over(ranking))
        .where(F.col("rank") <= k))


def als_top_k(model: ALSModel, history: DataFrame, limit: int, k: int) -> DataFrame:
    # Suréchantillonner garantit qu'il reste K candidats si le catalogue le permet.
    candidates = (model.recommendForAllUsers(limit)
        .select("userId", F.explode("recommendations").alias("rec"))
        .select("userId", F.col("rec.movieId").alias("movieId"), F.col("rec.rating").alias("score")))
    return unseen_top_k(candidates, history, k)


def popularity_top_k(history: DataFrame, limit: int, k: int) -> DataFrame:
    popular = (history.groupBy("movieId")
        .agg(F.count("*").alias("score"))
        .orderBy(F.desc("score"), F.asc("movieId"))
        .limit(limit))
    users = history.select("userId").distinct()
    return unseen_top_k(users.crossJoin(F.broadcast(popular)), history, k)


def ranking_metrics(recommendations: DataFrame, relevant: DataFrame, k: int) -> dict:
    targets = relevant.groupBy("userId").agg(F.count("*").alias("n_relevant"))
    hits = (recommendations.join(relevant, ["userId", "movieId"], "inner")
        .groupBy("userId").agg(F.count("*").alias("hits")))
    per_user = targets.join(hits, "userId", "left").fillna(0, subset=["hits"])
    scores = per_user.agg(
        F.count("*").alias("users"),
        F.avg(F.col("hits") / k).alias("precision_at_k"),
        F.avg(F.col("hits") / F.col("n_relevant")).alias("recall_at_k"),
        F.avg((F.col("hits") > 0).cast("double")).alias("hit_rate_at_k"),
    ).first()
    return scores.asDict()


def evaluate(train: DataFrame, test: DataFrame, k: int) -> dict:
    relevant = test.where(F.col("rating") >= RELEVANT_RATING).select("userId", "movieId").cache()
    relevant_count = relevant.count()
    if relevant_count == 0:
        raise ValueError("Le test doit contenir au moins une note de 4/5 ou plus.")

    limit = candidate_limit(train, k)
    model = train_als(train)
    print("Calcul des métriques pour ALS et le classement par popularité…", flush=True)
    return {
        "k": k,
        "relevant_rating_threshold": RELEVANT_RATING,
        "test_fraction_per_user": TEST_FRACTION,
        "train_ratings": train.count(),
        "test_ratings": test.count(),
        "relevant_test_ratings": relevant_count,
        "relevant_ratings_for_movies_absent_from_train": relevant.join(
            train.select("movieId").distinct(), "movieId", "left_anti"
        ).count(),
        "als": ranking_metrics(als_top_k(model, train, limit, k), relevant, k),
        "popularity": ranking_metrics(popularity_top_k(train, limit, k), relevant, k),
    }


def write_results(recommendations: DataFrame, metrics: dict, output_dir: Path, user_id: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["userId", "rank", "movieId", "title", "genres", "score"]
    csv_path = output_dir / "recommendations.csv"
    temp_path = csv_path.with_suffix(".csv.tmp")
    example = []
    try:
        with temp_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for row in recommendations.select(*columns).orderBy("userId", "rank").toLocalIterator():
                writer.writerow(row)
                if row.userId == user_id:
                    example.append(row)
        temp_path.replace(csv_path)
    finally:
        temp_path.unlink(missing_ok=True)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nRésultats écrits dans {output_dir.resolve()}")
    for name in ("als", "popularity"):
        scores = metrics[name]
        print(f"{name:10s}  Précision@{metrics['k']} = {scores['precision_at_k']:.2%}"
              f"  Rappel@{metrics['k']} = {scores['recall_at_k']:.2%}"
              f"  Hit rate@{metrics['k']} = {scores['hit_rate_at_k']:.2%}")
    print(f"\nRecommandations pour l’utilisateur {user_id}")
    if not example:
        print("Aucun film à afficher pour cet identifiant.")
    for row in example:
        print(f"{row.rank:2d}. {row.title} ({row.score:.2f})")


def main() -> None:
    args = parse_args()
    for filename in ("ratings.csv", "movies.csv"):
        if not (args.data_dir / filename).is_file():
            raise SystemExit(f"Fichier manquant : {args.data_dir / filename}. Voir README.md.")

    configure_windows_temp()
    spark = (SparkSession.builder
        .appName("MovieLens PySpark")
        .master("local[4]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        ratings, movies = load_data(spark, args.data_dir)
        ratings = ratings.cache()
        train, test = split_ratings(ratings)
        train = train.cache()
        print("Entraînement et évaluation sur les notes récentes…", flush=True)
        metrics = evaluate(train, test, args.top_k)
        train.unpersist()

        print("Réentraînement sur toutes les notes pour produire les recommandations…", flush=True)
        model = train_als(ratings)
        limit = candidate_limit(ratings, args.top_k)
        recommendations = als_top_k(model, ratings, limit, args.top_k).join(movies, "movieId")
        write_results(recommendations, metrics, args.output_dir, args.user)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
