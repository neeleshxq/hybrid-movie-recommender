import pandas as pd

from src.config import (
    MOVIES_PATH,
    RATINGS_PATH,
    TRAIN_FRACTION,
    VAL_FRACTION,
)


def load_data():
    ratings = pd.read_csv(
        RATINGS_PATH,
        sep="::",
        engine="python",
        names=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ],
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64",
        },
    )

    movies = pd.read_csv(
        MOVIES_PATH,
        sep="::",
        engine="python",
        names=[
            "movieId",
            "title",
            "genres",
        ],
        encoding="latin-1",
        dtype={
            "movieId": "int32",
        },
    )

    ratings = ratings.sort_values(
        ["userId", "timestamp"]
    ).reset_index(drop=True)

    return ratings, movies


def temporal_split(ratings):
    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in ratings.groupby("userId", sort=False):
        n = len(group)

        if n < 3:
            train_parts.append(group)
            continue

        train_end = max(
            1,
            int(n * TRAIN_FRACTION),
        )

        val_end = max(
            train_end + 1,
            int(n * (TRAIN_FRACTION + VAL_FRACTION)),
        )

        if val_end >= n:
            val_end = n - 1

        train_parts.append(
            group.iloc[:train_end]
        )

        val_parts.append(
            group.iloc[train_end:val_end]
        )

        test_parts.append(
            group.iloc[val_end:]
        )

    train = pd.concat(
        train_parts,
        ignore_index=True,
    )

    val = (
        pd.concat(
            val_parts,
            ignore_index=True,
        )
        if val_parts
        else ratings.iloc[0:0].copy()
    )

    test = (
        pd.concat(
            test_parts,
            ignore_index=True,
        )
        if test_parts
        else ratings.iloc[0:0].copy()
    )

    return train, val, test