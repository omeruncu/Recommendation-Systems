#############################
# Content Based Recommendation
#############################

# Business Problem
## A newly established online movie watching platform wants to recommend movies to its users.
## Since the login rate of users is very low, it cannot collect user habits. For this reason, it cannot develop product recommendations with collaborative filtering methods.
## However, it knows which movies users have watched in their browser traces. Make movie recommendations according to this information

# Dataset Story: https://www.kaggle.com/rounakbanik/the-movies-dataset
## These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages.
## This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.

## This dataset consists of the following files:
## movies_metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.
## keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.
## credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.
## links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.
## links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.
## ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#############################
# Developing Recommendations Based on Movie Overviews
#############################

# 1. Creating TF-IDF Matrix
# 2. Creating Cosine Similarity Matrix
# 3. Making Recommendations Based on Similarities
# 4. Preparing the Working Script

#################################
# 1. Creating TF-IDF Matrix
#################################

df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning
df.head()
df.shape

df["overview"].head()

tfidf = TfidfVectorizer(stop_words="english")

# df[df['overview'].isnull()]
df['overview'] = df['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(df['overview'])

tfidf_matrix.shape

df['title'].shape

tfidf.get_feature_names_out()

tfidf_matrix.toarray()


#################################
# 2. Creating the Cosine Similarity Matrix
#################################

cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)

cosine_sim.shape
cosine_sim[1]


#################################
# 3. Making Recommendations Based on Similarities
#################################

indices = pd.Series(df.index, index=df['title'])

indices.index.value_counts()

indices = indices[~indices.index.duplicated(keep='last')]

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df['title'].iloc[movie_indices]

#################################
# 4. Preparation of Working Script
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # Creating indexes
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # Capture index of title
    movie_index = indices[title]
    # Calculating similarity scores by title
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # Don't bring the top 10 movies except for itself
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)