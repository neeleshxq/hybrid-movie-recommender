# MovieLens Hybrid Recommender

A lightweight hybrid movie recommendation system using collaborative filtering, TF-IDF content retrieval, and LightGBM Learning-to-Rank.

## Pipeline

MovieLens ratings → temporal split → SVD collaborative filtering + TF-IDF content retrieval → hybrid candidate generation → feature engineering → LightGBM LambdaRank → Top-K evaluation.

## Models

- Popularity baseline
- SVD collaborative filtering
- TF-IDF content-based retrieval
- Hybrid CF + content
- Hybrid + LightGBM LambdaRank

## Metrics

- Precision@10
- Recall@10
- NDCG@10

## Data

This project expects a MovieLens-format dataset with:

```text
data/ml-1m/ratings.dat
data/ml-1m/movies.dat
```

The code is compatible with the MovieLens `::`-separated format. If using a 1M subset created from MovieLens 25M, describe it publicly as a **1M-rating subset of MovieLens 25M**, not as the official MovieLens 1M benchmark.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Results are saved to `outputs/evaluation_results.csv`.
