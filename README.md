In this example project, we use MovieLens 25M Dataset (https://grouplens.org/datasets/movielens/25m/Links to an external site. ).

Part I: Basic Data Manipulation & Simple Recommendation

Read in the rating file and create an RDD consisting of parsed lines, then count the number of ratings.
Recommend 5 movies with the highest average rating.
Other operations to enrich your data analysis.
Try to create visualizations to convey the insights.
Part II: Rating Prediction

First split rating data into 70% training set and 30% testing set.
Choose one matrix factorization algorithm to predict the rating score based on the rating data file only.
Extract features from movies and users (join movie and user data and do some feature transformation), then build another machine learning model to predict rating scores for the testing set.
Compare the pros and cons of these two models and report it.
Try to create visualizations to convey the insights.
You are suggested to choose suitable machine learning models in MLlib to do this task and you can refer to https://spark.apache.org/docs/latest/ml-guide.html Links to an external site.for more help.
