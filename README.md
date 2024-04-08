In this project, we use MovieLens 25M Dataset (https://grouplens.org/datasets/movielens/25m/Links to an external site. ).

# Part I: Basic Data Manipulation & Simple Recommendation

1. Read in the rating file and create an RDD consisting of parsed lines, then count the number of ratings.
2. Recommend 5 movies with the highest average rating.
3. Other operations to enrich your data analysis.

# Part II: Rating Prediction

1. First split rating data into 70% training set and 30% testing set.
2. Choose one matrix factorization algorithm to predict the rating score based on the rating data file only.
3. Extract features from movies and users (join movie and user data and do some feature transformation), then build another machine learning model to predict rating scores for the testing set.
4. Compare the pros and cons of these two models and report it.

You are suggested to choose suitable machine learning models in MLlib to do this task and you can refer to https://spark.apache.org/docs/latest/ml-guide.html Links to an external site.for more help.

The  distributed spark environment is based on Google Dataproc (https://cloud.google.com/dataproc)
