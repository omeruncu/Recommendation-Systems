###########################################
# Item-Based Collaborative Filtering
######################################
from kiwisolver import Variable

# Business Problem:
## An online movie viewing platform wants to develop a recommendation system with a collaborative filtering method.
## The company, which is testing content-based recommendation systems, wants to develop recommendations that will accommodate the opinions of the community.
## When users like a movie, they want to recommend other movies that have a similar liking pattern to that movie.

# Dataset Story: https://grouplens.org/datasets/movielens/
## Dataset provided by movieLens
## Contains movies and their ratings
## Dataset contains approximately 2000000 ratings for 27000 movies.

# Variables :
## movie.csv
### movieId - Unique movie number (UniqueID) (Same as in rating.csv file)
### title - Movie name

## rating.csv
### userid - Unique user number (UniqueID)
### movieId - Unique movie number (UniqueID) (Same as in movie.csv file)
### rating - Rating given to the movie by the user
### timestamp - Rating date

# Step 1: Preparing the Dataset
# Step 2: Creating the User Movie Df
# Step 3: Making Item-Based Movie Recommendations
# Step 4: Preparing the Working Script

######################################
# Step 1: Preparing the Dataset
######################################
import pandas as pd
pd.set_option('display.max_columns', 500)

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

df = movie.merge(rating, how="left", on="movieId")
df.head()


######################################
# Step 2: Creating the User Movie Df
######################################

df.shape

df["title"].nunique()

df["title"].value_counts().head()

comment_counts = df["title"].value_counts().reset_index()

comment_counts.columns = ["title", "count"]

rare_movies = comment_counts[comment_counts["count"] <= 1000]["title"]

common_movies = df[~df["title"].isin(rare_movies)]
common_movies["title"].nunique()
df["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
user_movie_df.columns


######################################
# Step 3: Making Item-Based Movie Recommendations
######################################

movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Insomnia", user_movie_df)


######################################
# Step 4: Preparing the Working Script
######################################
# Memory Scope:
## Variables defined inside the function do not spill out of the function. This prevents the global namespace from getting dirty and prevents unnecessary variables from remaining in memory for a long time.
## Related tasks are grouped under a clear name, which simplifies maintenance and increases testability.
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = df["title"].value_counts().reset_index()
    comment_counts.columns = ["title", "count"]
    rare_movies = comment_counts[comment_counts["count"] <= 1000]["title"]
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)
