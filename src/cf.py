import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.config import (
    CF_CANDIDATES,
    RANDOM_STATE,
    SVD_COMPONENTS,
    SVD_ITERATIONS,
)


class CFModel:
    def __init__(self, train):
        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.idx_to_movie = None

        self.user_factors = None
        self.movie_factors = None

        self.seen_by_user = {}

        users = train["userId"].unique()
        movies = train["movieId"].unique()

        self.user_to_idx = {
            int(user_id): index
            for index, user_id in enumerate(users)
        }

        self.movie_to_idx = {
            int(movie_id): index
            for index, movie_id in enumerate(movies)
        }

        self.idx_to_movie = np.asarray(movies)

        rows = (
            train["userId"]
            .map(self.user_to_idx)
            .to_numpy()
        )

        cols = (
            train["movieId"]
            .map(self.movie_to_idx)
            .to_numpy()
        )

        values = train["rating"].to_numpy(
            dtype=np.float32
        )

        matrix = csr_matrix(
            (
                values,
                (rows, cols),
            ),
            shape=(
                len(users),
                len(movies),
            ),
            dtype=np.float32,
        )

        n_components = min(
            SVD_COMPONENTS,
            matrix.shape[1] - 1,
        )

        if n_components < 1:
            raise ValueError(
                "Not enough movies for SVD."
            )

        print(
            f"CF matrix: {matrix.shape} | "
            f"SVD components: {n_components}"
        )

        svd = TruncatedSVD(
            n_components=n_components,
            n_iter=SVD_ITERATIONS,
            random_state=RANDOM_STATE,
        )

        self.user_factors = svd.fit_transform(
            matrix
        )

        self.movie_factors = svd.components_.T

        self.seen_by_user = (
            train.groupby("userId")["movieId"]
            .apply(
                lambda x: set(
                    x.astype(int)
                )
            )
            .to_dict()
        )

    def get_scores(
        self,
        user_id,
        top_n=CF_CANDIDATES,
    ):
        if int(user_id) not in self.user_to_idx:
            return {}

        user_idx = self.user_to_idx[
            int(user_id)
        ]

        scores = (
            self.user_factors[user_idx]
            @ self.movie_factors.T
        )

        seen = self.seen_by_user.get(
            int(user_id),
            set(),
        )

        if seen:
            seen_indices = [
                self.movie_to_idx[movie_id]
                for movie_id in seen
                if movie_id in self.movie_to_idx
            ]

            scores[seen_indices] = -np.inf

        n = min(top_n, len(scores))

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
            int(self.idx_to_movie[index]): float(
                scores[index]
            )
            for index in indices
            if np.isfinite(scores[index])
        }