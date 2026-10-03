import numpy as np
import pandas as pd


def minmax(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return values

    low = np.min(values)
    high = np.max(values)

    if high - low < 1e-12:
        return np.zeros_like(values)

    return (
        (values - low)
        / (high - low)
    )


def generate_candidates(
    user_id,
    history,
    cf_model,
    content_model,
):
    cf_scores = cf_model.get_scores(
        user_id
    )

    content_scores = content_model.get_scores(
        history
    )

    movie_ids = (
        set(cf_scores)
        | set(content_scores)
    )

    rows = []

    for movie_id in movie_ids:
        cf_score = cf_scores.get(
            movie_id,
            0.0,
        )

        content_score = content_scores.get(
            movie_id,
            0.0,
        )

        rows.append(
            {
                "userId": int(user_id),
                "movieId": int(movie_id),
                "cf_score": cf_score,
                "content_score": content_score,
                "from_cf": int(
                    movie_id in cf_scores
                ),
                "from_content": int(
                    movie_id in content_scores
                ),
            }
        )

    candidates = pd.DataFrame(rows)

    if candidates.empty:
        return candidates

    candidates["hybrid_score"] = (
        0.5
        * minmax(
            candidates["cf_score"].to_numpy()
        )
        + 0.5
        * minmax(
            candidates["content_score"].to_numpy()
        )
    )

    return (
        candidates
        .sort_values(
            "hybrid_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )