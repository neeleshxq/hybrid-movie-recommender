import pandas as pd


def build_stats(train):
    movie_stats = (
        train.groupby("movieId")["rating"]
        .agg(
            movie_mean_rating="mean",
            movie_rating_count="count",
        )
        .reset_index()
    )

    user_stats = (
        train.groupby("userId")["rating"]
        .agg(
            user_mean_rating="mean",
            user_rating_count="count",
        )
        .reset_index()
    )

    return movie_stats, user_stats


def build_features(
    candidates,
    train,
    movies,
    movie_stats,
    user_stats,
):
    features = candidates.copy()

    features = features.merge(
        movie_stats,
        on="movieId",
        how="left",
    )

    features = features.merge(
        user_stats,
        on="userId",
        how="left",
    )

    genre_counts = movies[
        ["movieId", "genres"]
    ].copy()

    genre_counts["genre_count"] = (
        genre_counts["genres"]
        .fillna("")
        .str.split("|")
        .apply(
            lambda genres: (
                0
                if genres == [""]
                else len(genres)
            )
        )
    )

    features = features.merge(
        genre_counts[
            ["movieId", "genre_count"]
        ],
        on="movieId",
        how="left",
    )

    features = features.fillna(0)

    return features