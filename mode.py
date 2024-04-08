import pandas as pd

df = pd.read_csv('../data/ml-25m/ratings.csv')
df.drop('timestamp', axis=1, inplace=True)
movie_mode = df.groupby('movieId')['rating'].apply(lambda x: x.mode()[0]).reset_index()
user_mode = df.groupby('userId')['rating'].apply(lambda x: x.mode()[0]).reset_index()

# 可选：重命名列名
movie_mode = movie_mode.rename(columns={'rating': 'mode_movie'})
user_mode = user_mode.rename(columns={'rating': 'mode_user'})

df = pd.merge(df, movie_mode, on='movieId', how='left')
df = pd.merge(df, user_mode, on='userId', how='left')
print(df)

df.to_csv('../data/ml-25m/ratings_mode.csv')
