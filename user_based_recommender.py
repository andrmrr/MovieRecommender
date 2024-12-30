import pandas as pd
import numpy as np
import time

import similarity as sim
import naive_recommender as nav
import utils as ut


def generate_m(movies_idx, users, ratings):
    m = pd.DataFrame(index=users, columns=movies_idx)
    for user in users:
        user_ratings = ratings[ratings["userId"] == user]
        for _, rating in user_ratings.iterrows():
            m.at[user, rating["movieId"]] = rating["rating"]    
    return m 


def user_based_recommender(target_user_idx, matrix):
    # Get the target user's ratings as a Series and a list
    target_user = matrix.loc[target_user_idx]
    target_user_vector = target_user.to_list()

    # Compute the similarity between the target user and each other user in the matrix.
    print("Computing the similarity vector for user {}...".format(target_user_idx))
    sims = pd.DataFrame(columns=["userId", "similarity"])
    sims.reindex(matrix.index)
    sims["similarity"] = matrix.apply(lambda row: sim.compute_similarity(target_user_vector, row.to_list()), axis=1)
    sims.userId = matrix.index
    sims = sims[sims.userId != target_user_idx]
    
    # Determine the unseen movies by the target user
    unseen_movies = target_user[target_user.isna()].index

    # Find the top U most similar users to the target user
    print("Finding nearest neighbors for user {}...".format(target_user_idx))
    U = 10 # Number of users to consider
    top_users_df = sims.sort_values("similarity", ascending=False).head(U)
    top_users = top_users_df["userId"].to_list()
    top_users_sim = top_users_df["similarity"].to_list()
    sum_sims = sum(top_users_sim)
    top_users_sim = [sim/sum_sims for sim in top_users_sim]

    # Compute the average rating for the target user and each of the top U users
    target_nonan = [x for x in target_user if not np.isnan(x)]
    target_avg = sum(target_nonan) / len(target_nonan)
    user_avgs = []
    for i in range(U):
        uid = top_users[i]
        user = matrix.loc[uid]
        user_nonan = [x for x in user if not np.isnan(x)]
        user_avg = sum(user_nonan) / len(user_nonan)
        user_avgs.append(user_avg)

    # Generate recommendations for unrated movies based on user similarity and ratings.
    recommendations = []
    print("Predicting ratings for user {}...".format(target_user_idx))
    for movie in unseen_movies:
        if movie in matrix.columns:
            prediction = target_avg
            for i in range(U):
                user = matrix.loc[top_users[i]]
                if not np.isnan(user[movie]):
                    prediction += top_users_sim[i] * (user[movie] - user_avgs[i])
            recommendations.append((movie, prediction))
           
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations



if __name__ == "__main__":
    start = time.time()

    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = ut.split_users(dataset["ratings.csv"], val_movies)

    # Create matrix between user and movies 
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))
    m = generate_m(movies_idx, users_idy, ratings_train)

    # user-to-user similarity
    target_user_idx = 123
    recommendations = user_based_recommender(target_user_idx, m)
    print(recommendations[:10])
     
    # The following code print the top 5 recommended films to the user
    print("\nTop 5 recommendations for user {}".format(target_user_idx))
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print ("{} --- (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))
    
    print("\nExecution time: ", round(time.time() - start), " seconds.")
