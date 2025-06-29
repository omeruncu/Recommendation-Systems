#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install scikit-surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Step 1: Dataset Preparation
# Step 2: Modeling
# Step 3: Model Tuning
# Step 4: Final Model and Estimation



#############################
# Step 1: Dataset Preparation
#############################
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

df = movie.merge(rating, how="left", on="movieId")
df.head()

df.shape

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

##############################
# Step 2: Modeling
##############################

trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)

accuracy.rmse(predictions)


svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)


sample_df[sample_df["userId"] == 1]

##############################
# Step 3: Model Tuning
##############################

param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}


gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

gs.fit(data)

gs.best_score['rmse'] # 0.930158639264083
gs.best_params['rmse'] # {'n_epochs': 5, 'lr_all': 0.002}


param_grid = {
    'n_factors': [20, 50, 100],
    'n_epochs': [10, 20, 30],
    'lr_all': [0.002, 0.005, 0.007],
    'reg_all': [0.02, 0.05, 0.1]
}

gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

gs.fit(data)

gs.best_score['rmse'] # 0.9307528070657397
gs.best_params['rmse'] # {'n_factors': 50, 'n_epochs': 10, 'lr_all': 0.007, 'reg_all': 0.1}

##############################
# Step 4: Final Model and Estimation
##############################

dir(svd_model)
svd_model.n_epochs

svd_model = SVD(**gs.best_params['rmse'])

data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)