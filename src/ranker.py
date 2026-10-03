import lightgbm as lgb
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    LGBM_ESTIMATORS,
    LGBM_LEARNING_RATE,
    LGBM_NUM_LEAVES,
    MAX_TRAIN_USERS,
)
from src.features import build_features
from src.retrieval import generate_candidates


def create_ranking_dataset(
    train,
    val,
    movies,
    cf_model,
    content_model,
    movie_stats,
    user_stats,
):
    user_groups = train.groupby(
        "userId",
        sort=False,
    )

    validation_lookup = {}

    for user_id, group in val.groupby(
        "userId"
    ):
        validation_lookup[int(user_id)] = {
            int(row.movieId): float(row.rating)
            for row in group.itertuples()
        }

    frames = []
    groups = []

    users = list(
        validation_lookup.keys()
    )[:MAX_TRAIN_USERS]

    print(
        f"Creating LightGBM ranking dataset "
        f"for {len(users)} users..."
    )

    for user_id in users:
        try:
            history = user_groups.get_group(
                user_id
            )
        except KeyError:
            continue

        candidates = generate_candidates(
            user_id,
            history,
            cf_model,
            content_model,
        )

        if candidates.empty:
            continue

        features = build_features(
            candidates,
            train,
            movies,
            movie_stats,
            user_stats,
        )

        ratings = validation_lookup[user_id]

        labels = []

        for movie_id in features["movieId"]:
            rating = ratings.get(
                int(movie_id),
                0,
            )

            if rating >= 5:
                labels.append(4)
            elif rating >= 4:
                labels.append(3)
            elif rating >= 3:
                labels.append(2)
            else:
                labels.append(0)

        features["label"] = labels

        frames.append(features)
        groups.append(len(features))

    if not frames:
        raise RuntimeError(
            "No ranking data was created."
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    return data, groups


class LTRModel:
    def __init__(self):
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=LGBM_ESTIMATORS,
            learning_rate=LGBM_LEARNING_RATE,
            num_leaves=LGBM_NUM_LEAVES,
            random_state=42,
            verbosity=-1,
        )

    def fit(self, data, groups):
        self.model.fit(
            data[FEATURE_COLUMNS],
            data["label"],
            group=groups,
        )

    def recommend(
        self,
        user_id,
        history,
        train,
        movies,
        cf_model,
        content_model,
        movie_stats,
        user_stats,
        top_k,
    ):
        candidates = generate_candidates(
            user_id,
            history,
            cf_model,
            content_model,
        )

        if candidates.empty:
            return []

        features = build_features(
            candidates,
            train,
            movies,
            movie_stats,
            user_stats,
        )

        features["ltr_score"] = (
            self.model.predict(
                features[FEATURE_COLUMNS]
            )
        )

        return (
            features
            .sort_values(
                "ltr_score",
                ascending=False,
            )
            .head(top_k)["movieId"]
            .astype(int)
            .tolist()
        )