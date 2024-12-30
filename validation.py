import random
from matplotlib import pyplot as plt
import utils
from user_based_recommender import user_based_recommender, generate_m
import numpy as np
from naive_recommender import naive_recommender

def genre_distribution(movies_list, genre_matrix):
    """
    Calculate genre frequencies for a list of movies.
    :param movies_list: List of movie IDs.
    :param genre_matrix: Matrix containing movie-genre relationships.
    :return: A dictionary with genre frequencies.
    """
    genres = genre_matrix.loc[movies_list].sum()
    total = genres.sum()
    genre_freq = genres / total
    return genre_freq

def validate_genre_similarity(target_user, ratings, movies, k=10):
    """
    Compare the genre resemblance of the top-k movies from two recommenders
    with the validation set for a target user.
    """
    # Step 1: Load the genre matrix
    genre_matrix = utils.matrix_genres(movies)

    # Step 2: Split validation and training sets
    val_movies = 5
    ratings_train, ratings_val = utils.split_users(ratings, val_movies)
    user_validation_set = ratings_val[ratings_val['userId'] == target_user]['movieId'].tolist()
    
    # Create matrix between user and movies 
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))
    m = generate_m(movies_idx, users_idy, ratings_train)

    # Step 3: Get top-k recommendations
    naive_recommendations = naive_recommender(ratings_train, movies, k)
    user_based_recommendations = user_based_recommender(target_user, m)
    user_based_recommendations = [id for id, _ in user_based_recommendations]
    
    # Step 4: Calculate genre distributions
    validation_genre_freq = genre_distribution(user_validation_set, genre_matrix)
    naive_genre_freq = genre_distribution(naive_recommendations, genre_matrix)
    user_based_genre_freq = genre_distribution(user_based_recommendations[:k], genre_matrix)
    
    # Step 5: Compare genre distributions (e.g., using L2-norm)
    naive_similarity = np.linalg.norm(validation_genre_freq - naive_genre_freq)
    user_based_similarity = np.linalg.norm(validation_genre_freq - user_based_genre_freq)
    
    # Step 6: Output results
    print(f"User: {target_user}")
    print(f"Naive Recommender Genre Distance: {naive_similarity}")
    print(f"User-Based Recommender Genre Distance: {user_based_similarity}")

    return naive_similarity, user_based_similarity

if __name__ == "__main__":
    path_to_ml_latest_small = "./ml-latest-small/"
    dataset = utils.load_dataset_from_source(path_to_ml_latest_small)
    ratings, movies = dataset["ratings.csv"], dataset["movies.csv"]
    user_ids = list(set(ratings['userId']))
    target_user_ids = random.sample(user_ids, 40)
    l2_distances= []
    for user in target_user_ids:
        val1, val2 = validate_genre_similarity(user, ratings, movies, k=10)
        l2_distances.append((val1, val2))
    naive_distances = [t[0] for t in l2_distances] 
    user_distances = [t[1] for t in l2_distances]
    indices = range(len(l2_distances)) 
    plt.figure(figsize=(8, 5))
    plt.plot(indices, naive_distances, marker='o', label='Naive L2 distances')
    plt.plot(indices, user_distances, marker='s', label='User-to-user L2 distances')
    plt.xlabel("Index")
    plt.ylabel("L2 Distance")
    plt.title("Comparison of the recommender systems")
    plt.legend()
    plt.show()
    avg_l2distance_naive = np.average(naive_distances)
    avg_l2distance_user = np.average(user_distances)
    print(f'Average L2 distance of the naive recommender system is {avg_l2distance_naive}')
    print(f'Average L2 distance of the user-to-user recommender system is {avg_l2distance_user}')