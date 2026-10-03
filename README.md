# MovieLens 1M Recommender System

A hybrid movie recommendation system built on the official **MovieLens 1M** dataset. The project combines collaborative filtering, content-based retrieval, hybrid candidate generation, and Learning-to-Rank to generate personalized Top-K recommendations.

## Pipeline

```text
MovieLens 1M
     │
     ▼
Temporal Train / Validation / Test Split
     │
     ├──────────────► Collaborative Filtering
     │                 Truncated SVD
     │
     └──────────────► Content Retrieval
                       TF-IDF
     │
     ▼
Hybrid Candidate Generation
     │
     ▼
Feature Engineering
     │
     ▼
LightGBM Learning-to-Rank
     │
     ▼
Top-K Recommendations
     │
     ▼
Precision@K / Recall@K / NDCG@K
```

## Key Features

* Temporal train/validation/test splitting to avoid using future interactions for training.
* **Collaborative filtering** using Truncated SVD on the user-item rating matrix.
* **Content-based retrieval** using TF-IDF representations of movie titles and genres.
* Hybrid candidate generation from collaborative and content-based retrieval.
* Feature engineering using:

  * Collaborative filtering score
  * Content similarity score
  * Candidate-source indicators
  * Movie rating statistics
  * User rating statistics
  * Genre count
* **LightGBM LambdaRank** model for personalized candidate ranking.
* Comparison against popularity, collaborative filtering, content-based, and hybrid baselines.
* Evaluation using Precision@10, Recall@10, and NDCG@10.

## Dataset

This project uses the official **MovieLens 1M** dataset released by GroupLens.

The dataset contains approximately:

* 1 million ratings
* 6,000 users
* 4,000 movies
* Ratings from 1 to 5

Place the following files inside the `data/` directory:

```text
data/
├── ratings.dat
├── movies.dat
└── users.dat
```

The `.dat` files are read directly by the project; no conversion to CSV is required.

## Project Structure

```text
movie_recommender_ml1m/
│
├── data/
│   ├── ratings.dat
│   ├── movies.dat
│   └── users.dat
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── cf.py
│   ├── content.py
│   ├── features.py
│   ├── retrieval.py
│   ├── baselines.py
│   ├── ranker.py
│   └── evaluation.py
│
├── outputs/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Python 3.11 is recommended.

## Run

From the project root:

```bash
python main.py
```

The pipeline will:

1. Load the MovieLens 1M dataset.
2. Create a temporal train/validation/test split.
3. Train the SVD collaborative filtering model.
4. Build the TF-IDF content model.
5. Generate retrieval candidates.
6. Build ranking features.
7. Train the LightGBM ranker.
8. Evaluate all recommendation approaches.
9. Save the evaluation results.

Results are saved to:

```text
outputs/evaluation_results.csv
```

## Evaluation

The system compares five recommendation approaches:

| Model                   | Description                                  |
| ----------------------- | -------------------------------------------- |
| Popularity              | Rating-based popularity baseline             |
| Collaborative Filtering | Truncated SVD retrieval                      |
| Content                 | TF-IDF content-based retrieval               |
| Hybrid                  | Combined CF + content retrieval              |
| Hybrid + LightGBM       | Hybrid candidates re-ranked using LambdaRank |

Evaluation metrics:

* **Precision@10** — fraction of the Top-10 recommendations that are relevant.
* **Recall@10** — fraction of relevant test items retrieved in the Top-10.
* **NDCG@10** — measures both relevance and ranking position.

## Technical Stack

**Python · Pandas · NumPy · SciPy · Scikit-learn · LightGBM**

## Why Learning-to-Rank?

Collaborative filtering and content retrieval provide complementary signals, but their raw scores do not directly capture all user-item relationships.

The LightGBM ranker combines retrieval scores with user, movie, and candidate features to learn how candidate movies should be ordered for each user.

## Reproducibility

The project uses a fixed random seed for reproducible model behavior:

```text
Random State: 42
```

The train/validation/test split is chronological for each user, ensuring that validation and test interactions occur after the interactions used for training.

## Limitations

This is a portfolio/research implementation rather than a production-scale recommendation service.

The current implementation limits the number of users used for ranking-model training and evaluation to keep the end-to-end experiment computationally manageable.

Scaling the same approach to a much larger dataset would require engineering improvements such as batched retrieval, precomputed features, more efficient candidate generation, and potentially approximate nearest-neighbor retrieval.

## Future Improvements

* Batch candidate generation and evaluation.
* Approximate nearest-neighbor retrieval for larger catalogs.
* More advanced ranking features.
* Hyperparameter tuning for retrieval and ranking models.
* Online recommendation serving and monitoring.

## License

This project is intended for educational and portfolio purposes.
