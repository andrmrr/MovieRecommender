import pandas as pd 
import utils as ut

def naive_recommender(ratings: object, movies:object, k: int = 10) -> list: 
    # Provide the code for the naive recommender here. This function should return 
    # the list of the top most viewed films according to the ranking (sorted in descending order).
    # Consider using the utility functions from the pandas library.

    movie_ratings = ratings.groupby("movieId").agg(avg=("rating", "mean"), count=("rating", "count"))
    movie_ratings = movie_ratings.sort_values(by=["avg", "count"], ascending=False)

    most_seen_movies = []
    for i in range(k):
        mov = movie_ratings.index[i]
        most_seen_movies.append(movies[movies["movieId"] == mov].iloc[0]["title"])
    
    return most_seen_movies


if __name__ == "__main__":
    
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)
    
    ratings, movies = dataset["ratings.csv"], dataset["movies.csv"]
    recom = naive_recommender(ratings, movies)
    for r in recom:
        print(r)

