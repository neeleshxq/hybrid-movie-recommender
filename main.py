from src.config import (
    MAX_EVAL_USERS,
    OUTPUT_DIR,
    TOP_K,
)
from src.content import ContentModel
from src.cf import CFModel
from src.data import (
    load_data,
    temporal_split,
)
from src.features import build_stats
from src.ranker import (
    LTRModel,
    create_ranking_dataset,
)


def main():
    print("Loading MovieLens dataset...")

    ratings, movies = load_data()

    print(
        "\nCreating temporal "
        "train/validation/test split..."
    )

    train, val, test = temporal_split(
        ratings
    )

    print(f"Train: {len(train):,}")
    print(f"Validation: {len(val):,}")
    print(f"Test: {len(test):,}")

    print(
        "\nTraining sparse SVD "
        "collaborative filtering..."
    )

    cf_model = CFModel(train)

    print(
        "\nBuilding TF-IDF content model..."
    )

    content_model = ContentModel(
        movies
    )

    print(
        f"TF-IDF matrix: "
        f"{content_model.matrix.shape}"
    )

    movie_stats, user_stats = build_stats(
        train
    )

    ranking_data, groups = (
        create_ranking_dataset(
            train,
            val,
            movies,
            cf_model,
            content_model,
            movie_stats,
            user_stats,
        )
    )

    print(
        "\nTraining LightGBM ranker..."
    )

    ltr_model = LTRModel()

    ltr_model.fit(
        ranking_data,
        groups,
    )

    print(
        "\nEvaluating recommenders..."
    )

    from src.evaluation import evaluate_all

    results = evaluate_all(
        train,
        test,
        movies,
        cf_model,
        content_model,
        ltr_model,
        movie_stats,
        user_stats,
        TOP_K,
        MAX_EVAL_USERS,
    )

    print("\nResults:")
    print(
        results.to_string(
            index=False
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_DIR
        / "evaluation_results.csv",
        index=False,
    )

    print(
        "\nSaved results to: "
        f"{OUTPUT_DIR / 'evaluation_results.csv'}"
    )


if __name__ == "__main__":
    main()