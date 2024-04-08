from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import rand, explode, split, when, avg, col
from pyspark.sql.types import FloatType
from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
import time

start_time = time.time()
# Increase heap memory size
conf = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "8g")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.appName('part2_1n2').getOrCreate()

# Load Dataframe
df = spark.read.csv("../data/ml-25m/ratings.csv",header=True)
df = df.select(df.userId.cast("integer"), df.movieId.cast("integer"), df.rating.cast("double"))

# Shuffle the data and divide it into training and testing sets
df = df.orderBy(rand())
(training, test) = df.randomSplit([0.7, 0.3])

# Use Alternating Least Squares
als = ALS(maxIter=15, regParam=0.01, userCol="movieId", itemCol="userId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# # Adjusting the hyperparameters of the ALS algorithm by setting the parameter grid
# paramGrid = ParamGridBuilder() \
#     .addGrid(als.rank, [10, 20, 30]) \
#     .addGrid(als.maxIter, [10, 20, 30]) \
#     .addGrid(als.regParam, [0.01, 0.05, 0.1]) \
#     .build()

# # Using TrainValidationSplit for model training and selecting the best parameters
# evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
# tvs = TrainValidationSplit(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator)
# model = tvs.fit(training)

# Predict rating by trained model
predictions = model.transform(test)

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