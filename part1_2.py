from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, round
import time
start_time = time.time()
# Create SparkContext
sc = SparkContext("top5movie")

# Create SparkSession from SparkContext
spark = SparkSession(sc)

# Read the rating file
rating_lines = sc.textFile("../data/ml-25m/ratings.csv")

# Skip the header line in rating file
rating_header = rating_lines.first()
rating_lines = rating_lines.filter(lambda line: line != rating_header)

# Parse the rating lines and create an RDD
rating_ratings = rating_lines.map(lambda line: line.split(","))

# Map the movieId and rating as key-value pairs
movie_ratings = rating_ratings.map(lambda x: (x[1], float(x[2])))

# Calculate the average rating and count the number of raters for each movieId
movie_ratings_count = movie_ratings.groupByKey().mapValues(lambda x: (sum(x) / len(x), len(x)))

# Read the movie file
movie_lines = sc.textFile("../data/ml-25m/movies.csv")

# Skip the header line in movie file
movie_header = movie_lines.first()
movie_lines = movie_lines.filter(lambda line: line != movie_header)

# Parse the movie lines and create an RDD
movies = movie_lines.map(lambda line: line.split(","))

# Map the movieId and title as key-value pairs
movie_titles = movies.map(lambda x: (x[0], x[1]))

# Join the average ratings RDD with the movie titles RDD
avg_ratings_with_titles = movie_ratings_count.join(movie_titles)

# Get the top 5 movies with the highest average ratings and more than a given number of raters
min_raters = 3000  # Set the minimum number of raters
top_ratings = avg_ratings_with_titles.filter(lambda x: x[1][0][1] > min_raters).takeOrdered(5, key=lambda x: -x[1][0][0])

# Map the RDD and then convert it to DataFrame
top_ratings = [(x[0], x[1][1], x[1][0][0], x[1][0][1]) for x in top_ratings]
top_ratings_df = spark.createDataFrame(top_ratings, ["MovieId", "Title", "AverageRating", "Number of Raters"])

# Remove the quotes in title and approximate rating number
top_ratings_df = top_ratings_df.withColumn("Title", regexp_replace("Title", '"', ''))
top_ratings_df =top_ratings_df.withColumn("AverageRating", round(top_ratings_df["AverageRating"], 2))
top_ratings_df.show(truncate=False)

# Write DataFrame to CSV
top_ratings_df.coalesce(1).write.mode("overwrite").csv(f"../result/top5_ratings_{min_raters}raters", header=True, encoding= "utf-8")

# Stop SparkContext
sc.stop()

end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "seconds")