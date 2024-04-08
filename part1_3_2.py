from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import seaborn as sns

# Create SparkSession
spark = SparkSession.builder.appName('genre_pop').getOrCreate()

df_rating = spark.read.csv('gs://project5021/data/ml-25m/ratings.csv', header=True, inferSchema=True)
df_movie = spark.read.csv('gs://project5021/data/ml-25m/movies.csv', header=True, inferSchema=True)

# Calculate the number of ratings for each movie
df_rating_count = df_rating.groupBy('movieId').count().orderBy('movieId')

# Split and expand the genres column
df_movie = df_movie.withColumn('genre', explode(split(df_movie['genres'], '[|]'))).drop('genres').drop('title')

# Group according to genes and calculate the quantity of each genre
df_genre_count = df_movie.groupBy('genre').count().orderBy('count', ascending=False)

# Connect df_rating_count on movieId
df_movie = df_movie.join(df_rating_count, on='movieId')

# Group genre and then sum count
df_rating_count = df_movie.groupBy('genre').agg(sum('count').alias('total_ratings')).orderBy('total_ratings', ascending=False)

# Merge two Dataframes and calculate the popularity
df_rating_count = df_rating_count.join(df_genre_count, on='genre')
df_rating_count = df_rating_count.withColumn('pop', round(df_rating_count['total_ratings'] / df_rating_count['count']).cast("integer"))
df_rating_count = df_rating_count.filter(df_rating_count['genre'] != '(no genres listed)').orderBy('pop', ascending=False)
df_rating_count.show()
df_rating_count.coalesce(1).write.mode("overwrite").csv("gs://project5021/result/popular_genre", header=True, encoding= "utf-8")


# Convert to Pandas DataFrame
df_rating_count_pandas = df_rating_count.toPandas()

# Draw the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='pop', y='genre', data=df_rating_count_pandas, orient='h')
plt.title('Popularity of Genres')
plt.xlabel('Popularity')
plt.ylabel('Genre')
plt.show()
plt.savefig('gs://project5021/result/genre_pop.png')

spark.stop()
