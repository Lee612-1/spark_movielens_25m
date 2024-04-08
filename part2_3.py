from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import udf, explode, split, when, avg, col
from pyspark.sql.types import FloatType
from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.feature import VectorAssembler
import time

start_time = time.time()

# Increase heap memory size
# conf = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "8g")
# sc = SparkContext(conf=conf)
spark = SparkSession.builder.master('local').appName('part2_3').getOrCreate()

# Load Dataframe
df_rating = spark.read.csv("../data/ml-25m/ratings_mode.csv", header=True)
df_rating = df_rating.select(df_rating.userId.cast("integer"), df_rating.movieId.cast("integer"), df_rating.rating.cast("double"), df_rating.mode_movie.cast("double"), df_rating.mode_user.cast("double"))
df_movie = spark.read.csv("../data/ml-25m/movies.csv", header=True)

# Calculate the average rating for each movieId
average_movie_ratings = df_rating.groupBy("movieId").agg(avg("rating").alias("movie_average"))
average_user_ratings = df_rating.groupBy("userId").agg(avg("rating").alias("user_average"))
df_rating = df_rating.join(average_movie_ratings, on='movieId', how='left')
df_rating = df_rating.join(average_user_ratings, on='userId', how='left')

# Process movie data
df_movie = df_movie.withColumn('genre', explode(split(df_movie['genres'], '[|]'))).drop('genres').drop('title')
df_movie = df_movie.groupBy("movieId").pivot("genre").count().fillna(0).drop("(no genres listed)")
df = df_rating.join(df_movie, on='movieId', how='left')

# Create feature vectors
assembler = VectorAssembler(
    inputCols=df.columns[3:],
    outputCol="features")

# Convert DataFrame to Feature Vector
df_vector = assembler.transform(df).select("features", "rating")

# Creating a Random Forest Regression Model
rf = RandomForestRegressor(featuresCol="features", labelCol="rating")
lr = LinearRegression(featuresCol="features", labelCol="rating")

# Shuffle the data and divide it into training and testing sets
(training, test) = df_vector.randomSplit([0.7, 0.3])

model_rf = rf.fit(training)

# Predict rating by trained model
predictions = model_rf.transform(test)

# Calculate the true prediction
# predictions = predictions.withColumn("roundedPrediction", when(col("prediction") < 3.4, 3.0)
#                    .when(col("prediction") < 4.0, 4.0)
#                    .otherwise(5.0))
predictions = predictions.withColumn("roundedPrediction", when(col("prediction") < 1.2 , 1.0)
                    .when(col("prediction") < 2.3, 2.0)
                    .when(col("prediction") < 3.4, 3.0)
                   .when(col("prediction") < 4.5, 4.0)
                   .otherwise(5.0))
# Calculate accuracy
correct_predictions = predictions.filter(predictions.rating == predictions.roundedPrediction).count()
total_predictions = predictions.count()
accuracy = correct_predictions / total_predictions

print("Accuracy: ", accuracy)

spark.stop()
end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "seconds")
