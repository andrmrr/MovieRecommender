import pandas as pd
import numpy as np

import similarity as sim
import naive_recommender as nav
import utils as ut


def generate_m(movies_idx, users, ratings):
    m = pd.DataFrame(index=users, columns=movies_idx)
    for user in users:
        user_ratings = ratings[ratings["userId"] == user]
        for _, rating in user_ratings.iterrows():
            m.loc[user][rating["movieId"]] = rating["rating"]
    
    return m 


def user_based_recommender(target_user_idx, matrix):
    target_user = matrix.loc[target_user_idx]
    # for i in range(300):
    #     print(target_user.to_list()[i], target_user.index.to_list()[i])
    # print(target_user.to_list()[225])
    target_user_vector = target_user.to_list()
    recommendations = []
    
    # Compute the similarity between  the target user and each other user in the matrix. 
    # We recommend to store the results in a dataframe (userId and Similarity)
    sims = pd.DataFrame(columns=["userId", "similarity"])
    sims.reindex(matrix.index)
    sims["similarity"] = matrix.apply(lambda row: sim.compute_similarity(target_user_vector, row.to_list()), axis=1)
    sims.userId = matrix.index
    sims = sims[sims.userId != target_user_idx]
    print(sims.head(20))

    
    # Determine the unseen movies by the target user. Those films are identfied 
    # since don't have any rating. 
    unseen_movies = target_user[target_user.isna()].index
    # print(unseen_movies[:300].tolist())

    # Find the top U most similar users to the target user
    U = 5 # Number of users to consider
    top_users_df = sims.sort_values("similarity", ascending=False).head(U)
    print(top_users_df)
    top_users = top_users_df["userId"].to_list()
    top_users_sim = top_users_df["similarity"].to_list()
    print(top_users, top_users_sim)
    sum_sims = sum(top_users_sim)
    top_users_sim = [sim/sum_sims for sim in top_users_sim]
    print(top_users, top_users_sim)
    # Generate recommendations for unrated movies based on user similarity and ratings.
    avg_rating = ???
    for movie in unseen_movies:
        if movie not in matrix.columns:
            for i in range(U):

        
    
    return recommendations



if __name__ == "__main__":
    
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
    # print(m.iloc[:, 40:].head(123))
     
    # The following code print the top 5 recommended films to the user
    # for recomendation in recommendations[:5]:
    #     rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
    #     print (" Recomendation :Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    
    # # Validation
    # matrixmpa_genres = ut.matrix_genres(dataset["movies.csv"])
    
     
    








