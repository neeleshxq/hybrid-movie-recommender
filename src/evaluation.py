import math

import pandas as pd

from src.baselines import (
    get_cf_recommendations,
    get_content_recommendations,
    get_hybrid_recommendations,
    popularity_scores,
)


def precision_at_k(
    recommended,
    relevant,
    k,
):
    if k == 0:
        return 0.0

    return (
        len(
            set(recommended[:k])
            & relevant
        )
        / k
    )


def recall_at_k(
    recommended,
    relevant,
    k,
):
    if not relevant:
        return 0.0

    return (
        len(
            set(recommended[:k])
            & relevant
        )
        / len(relevant)
    )


def ndcg_at_k(
    recommended,
    relevant,
    k,
):
    if not relevant:
        return 0.0

    hits = [
        1 if movie_id in relevant else 0
        for movie_id in recommended[:k]
    ]

    dcg = sum(
        hit / math.log2(index + 2)
        for index, hit in enumerate(hits)
    )

    ideal_hits = min(
        len(relevant),
        k,
    )

    idcg = sum(
        1 / math.log2(index + 2)
        for index in range(ideal_hits)
    )

    return (
        dcg / idcg
        if idcg
        else 0.0
    )


def evaluate_all(
    train,
    test,
    movies,
    cf_model,
    content_model,
    ltr_model,
    movie_stats,
    user_stats,
    top_k,
    max_users,
):
    train_groups = train.groupby(
        "userId",
        sort=False,
    )

    test_lookup = (
        test.groupby("userId")["movieId"]
        .apply(
            lambda x: set(
                x.astype(int)
            )
        )
        .to_dict()
    )

    users = (
    test["userId"]
    .drop_duplicates()
    .sample(
        n=min(max_users, test["userId"].nunique()),
        random_state=42,
    )
    .tolist()
)

    totals = {
        "Popularity": [0.0, 0.0, 0.0],
        "Collaborative Filtering": [
            0.0,
            0.0,
            0.0,
        ],
        "Content": [0.0, 0.0, 0.0],
        "Hybrid": [0.0, 0.0, 0.0],
        "Hybrid + LightGBM": [
            0.0,
            0.0,
            0.0,
        ],
    }

    count = 0

    for user_id in users:
        try:
            history = train_groups.get_group(
                user_id
            )
        except KeyError:
            continue

        relevant = test_lookup[user_id]

        popularity = (
            popularity_scores(
                train,
                history,
            )
            .head(top_k)["movieId"]
            .astype(int)
            .tolist()
        )

        collaborative = get_cf_recommendations(
            user_id,
            history,
            cf_model,
            top_k,
        )

        content = get_content_recommendations(
            history,
            content_model,
            top_k,
        )

        hybrid = get_hybrid_recommendations(
            user_id,
            history,
            cf_model,
            content_model,
            top_k,
        )

        lightgbm = ltr_model.recommend(
            user_id,
            history,
            train,
            movies,
            cf_model,
            content_model,
            movie_stats,
            user_stats,
            top_k,
        )

        methods = {
            "Popularity": popularity,
            "Collaborative Filtering": collaborative,
            "Content": content,
            "Hybrid": hybrid,
            "Hybrid + LightGBM": lightgbm,
        }

        for name, recommendations in methods.items():
            totals[name][0] += precision_at_k(
                recommendations,
                relevant,
                top_k,
            )

            totals[name][1] += recall_at_k(
                recommendations,
                relevant,
                top_k,
            )

            totals[name][2] += ndcg_at_k(
                recommendations,
                relevant,
                top_k,
            )

        count += 1

    results = []

    for name, values in totals.items():
        results.append(
            {
                "Model": name,
                "Precision@10": (
                    values[0] / count
                    if count
                    else 0
                ),
                "Recall@10": (
                    values[1] / count
                    if count
                    else 0
                ),
                "NDCG@10": (
                    values[2] / count
                    if count
                    else 0
                ),
            }
        )

    return pd.DataFrame(results)