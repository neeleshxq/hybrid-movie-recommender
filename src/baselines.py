import numpy as np
import pandas as pd

from src.retrieval import generate_candidates


def popularity_scores(train, history):
    seen = set(
        history["movieId"].astype(int)
    )

    stats = (
        train.groupby("movieId")["rating"]
        .agg(
            mean_rating="mean",
            rating_count="count",
        )
        .reset_index()
    )

    stats["score"] = (
        stats["mean_rating"]
        * np.log1p(
            stats["rating_count"]
        )
    )

    stats = stats[
        ~stats["movieId"].isin(seen)
    ]

    return stats.sort_values(
        "score",
        ascending=False,
    )


def get_cf_recommendations(
    user_id,
    history,
    cf_model,
    top_k,
):
    scores = cf_model.get_scores(
        user_id,
        top_k,
    )

    return list(scores)


def get_content_recommendations(
    history,
    content_model,
    top_k,
):
    scores = content_model.get_scores(
        history,
        top_k,
    )

    return list(scores)


def get_hybrid_recommendations(
    user_id,
    history,
    cf_model,
    content_model,
    top_k,
):
    candidates = generate_candidates(
        user_id,
        history,
        cf_model,
        content_model,
    )

    return (
        candidates.head(top_k)["movieId"]
        .astype(int)
        .tolist()
    )