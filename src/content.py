import numpy as np

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.config import (
    CONTENT_CANDIDATES,
    MAX_TFIDF_FEATURES,
    MIN_POSITIVE_RATING,
)


class ContentModel:
    def __init__(self, movies):
        self.movies = movies.reset_index(
            drop=True
        ).copy()

        self.movie_to_idx = {
            int(movie_id): index
            for index, movie_id in enumerate(
                self.movies["movieId"]
            )
        }

        text = (
            self.movies["title"].fillna("")
            + " "
            + self.movies["genres"]
            .fillna("")
            .str.replace(
                "|",
                " ",
                regex=False,
            )
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=MAX_TFIDF_FEATURES,
        )

        self.matrix = self.vectorizer.fit_transform(
            text
        )

        self.matrix = normalize(
            self.matrix
        )

    def get_scores(
        self,
        history,
        top_n=CONTENT_CANDIDATES,
    ):
        liked = history[
            history["rating"] >= MIN_POSITIVE_RATING
        ]

        if liked.empty:
            return {}

        rows = []
        weights = []

        for _, row in liked.iterrows():
            movie_id = int(row["movieId"])

            index = self.movie_to_idx.get(
                movie_id
            )

            if index is not None:
                rows.append(index)
                weights.append(
                    float(row["rating"])
                )

        if not rows:
            return {}

        profile = (
            self.matrix[rows]
            .multiply(
                np.asarray(weights)[:, None]
            )
            .sum(axis=0)
        )

        profile = np.asarray(profile)
        profile = normalize(profile)

        scores = cosine_similarity(
            profile,
            self.matrix,
        ).ravel()

        seen = set(
            history["movieId"].astype(int)
        )

        for movie_id in seen:
            index = self.movie_to_idx.get(
                movie_id
            )

            if index is not None:
                scores[index] = -np.inf

        n = min(
            top_n,
            len(scores),
        )

        if n <= 0:
            return {}

        indices = np.argpartition(
            scores,
            -n,
        )[-n:]

        indices = indices[
            np.argsort(
                scores[indices]
            )[::-1]
        ]

        return {
            int(
                self.movies.iloc[index]["movieId"]
            ): float(scores[index])
            for index in indices
            if np.isfinite(scores[index])
        }