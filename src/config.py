from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

RATINGS_PATH = DATA_DIR / "ratings.dat"
MOVIES_PATH = DATA_DIR / "movies.dat"

RANDOM_STATE = 42

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

SVD_COMPONENTS = 50
SVD_ITERATIONS = 5

CF_CANDIDATES = 100
CONTENT_CANDIDATES = 100

MIN_POSITIVE_RATING = 3.5
MAX_TFIDF_FEATURES = 20000

MAX_TRAIN_USERS = 1000
MAX_EVAL_USERS = 1000

LGBM_ESTIMATORS = 150
LGBM_LEARNING_RATE = 0.05
LGBM_NUM_LEAVES = 31

TOP_K = 10

FEATURE_COLUMNS = [
    "cf_score",
    "content_score",
    "from_cf",
    "from_content",
    "movie_mean_rating",
    "movie_rating_count",
    "user_mean_rating",
    "user_rating_count",
    "genre_count",
]
