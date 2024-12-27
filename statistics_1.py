import utils
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

if __name__ == "__main__":
    dataset_dict = utils.load_dataset_from_source("/Users/julianluneburg/Documents/tum-semesters/WS-24/IRRS/IRRS5/ml-latest-small") 
    ratings = dataset_dict['ratings.csv']['rating']
    average_movie_rating = np.round(np.average(ratings), 4)
    print(f'The average movie rating is {average_movie_rating}.')
    movie_rating_variance = np.round(np.var(ratings), 4)
    print(f'The movie rating variance is {movie_rating_variance}.')
    frequency = Counter(ratings)
    plt.bar(list(frequency.keys()), list(frequency.values()), color='skyblue', edgecolor='black', width=0.5)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Barplot of Rating Frequencies')
    plt.show()
    n_unique_users = len(set(dataset_dict['ratings.csv']['userId']))
    print(f'The number of unique users is {n_unique_users}.')
    avg_n_ratings_per_user = len(dataset_dict['ratings.csv']['userId'])/n_unique_users
    print(f'On average every user rated {avg_n_ratings_per_user} movies.')
    genre_count = {}
    for genres in dataset_dict['movies.csv']['genres']:
        for genre in genres.split('|'):
            if genre in genre_count:
                genre_count[genre] += 1
            else:
                genre_count[genre] = 1
    for genre, count in sorted(genre_count.items(), 
                               key=lambda item: item[1],
                               reverse=True):
        print(f'The genre {genre} appears {count} times.')