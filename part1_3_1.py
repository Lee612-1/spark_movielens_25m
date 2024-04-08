from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName('movie_trend').getOrCreate()

df = spark.read.csv('gs://project5021/data/ml-25m/ratings.csv', header=True, inferSchema=True)

# Calculate the number of ratings for each movie
df_rating_count = df.groupBy('movieId').count()

# Filter out movies with ratings less than 5000
df_rating_count = df_rating_count.filter(df_rating_count['count'] >= 5000)

# Only retain rating data for movies with ratings greater than or equal to 5000
df = df.join(df_rating_count, 'movieId')

# Convert timestamp to year
df = df.withColumn('Year', year(from_unixtime(df['timestamp'])))

# Calculate the total rating and number of ratings for each movie per year
df_yearly = df.groupBy('movieId', 'Year').agg(sum('rating').alias('yearly_total_rating'), count('rating').alias('yearly_count')).orderBy('movieid', 'Year')

# Define window functions to sort by movieId and year
window = Window.partitionBy('movieId').orderBy('Year')

# Calculate the cumulative total score and number of scores from the beginning to that year
df_yearly = df_yearly.withColumn('cumulative_total_rating', sum('yearly_total_rating').over(window))
df_yearly = df_yearly.withColumn('cumulative_count', sum('yearly_count').over(window))

# Calculate the cumulative average score from the beginning to that year
df_yearly = df_yearly.withColumn('cumulative_avg_rating', df_yearly['cumulative_total_rating'] / df_yearly['cumulative_count'])

# Starting from years with ratings greater than 100 people
df_yearly = df_yearly.filter(df_yearly["cumulative_count"] >= 100).orderBy('movieId', 'Year')

# Calculate the average rating for the earliest and latest years of each movie
df_first_rating = df_yearly.withColumn('first_rating', first('cumulative_avg_rating').over(window))
df_last_rating = df_yearly.withColumn('last_rating', last('cumulative_avg_rating').over(window))

# Calculate the rise and fall rate
df_diff = df_first_rating.join(df_last_rating, ['movieId', 'Year'])
df_diff = df_diff.withColumn('diff', (df_diff['last_rating'] - df_diff['first_rating']) / df_diff['first_rating'])

# Identify the movie with the highest growth rate
df_rising = df_diff.orderBy('diff', ascending=False).first()
df_rising_filtered = df_yearly.filter(df_yearly['movieId'] == df_rising['movieId'])
df_rising_filtered.coalesce(1).write.csv("gs://project5021/result/rating_rise", header=True, encoding= "utf-8")

# Identify the movie with the greatest decline
df_falling = df_diff.orderBy('diff', ascending=True).first()
df_falling_filtered = df_yearly.filter(df_yearly['movieId'] == df_falling['movieId'])
df_falling_filtered.coalesce(1).write.csv("gs://project5021/result/rating_fall", header=True, encoding= "utf-8")

# Find the corresponding movies' name
df_title = spark.read.csv('gs://project5021/data/ml-25m/movies.csv', header=True, inferSchema=True)
rising_title = df_title.filter(df_title.movieId == df_rising['movieId']).select('title').collect()[0][0]
falling_title = df_title.filter(df_title.movieId == df_falling['movieId']).select('title').collect()[0][0]
print('涨幅最大的电影:', df_rising['movieId'], rising_title)
print('跌幅最大的电影:', df_falling['movieId'], falling_title)

# Convert Spark DataFrame to Pandas DataFrame
df_rising_filtered = df_rising_filtered.toPandas()
df_falling_filtered = df_falling_filtered.toPandas()

# Draw a line chart of the movie with the highest increase
plt.figure(figsize=(10, 6))
plt.plot(df_rising_filtered['Year'], df_rising_filtered['cumulative_avg_rating'], marker='o', color='r')
for i, row in df_rising_filtered.iterrows():
    plt.text(row['Year'], row['cumulative_avg_rating'], f"{row['cumulative_avg_rating']:.2f}")
plt.title(f'Rating trend of \"{rising_title}\"')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.xticks(df_rising_filtered['Year'], rotation=90)
plt.savefig('gs://project5021/result/Rising_Movie.png')

#Draw a line chart of the movie with the largest drop
plt.figure(figsize=(10, 6))
plt.plot(df_falling_filtered['Year'], df_falling_filtered['cumulative_avg_rating'], marker='o', color='g')
for i, row in df_falling_filtered.iterrows():
    plt.text(row['Year'], row['cumulative_avg_rating'], f"{row['cumulative_avg_rating']:.2f}")
plt.title(f'Rating trend of \"{falling_title}\"')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.xticks(df_falling_filtered['Year'], rotation=90)
plt.savefig('gs://project5021/result/Falling_Movie.png')

spark.stop()
