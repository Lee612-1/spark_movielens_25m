#Part 1_3 汇总，比较速度

#1_3_1_趋势变化最大电影

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import time
start_time = time.time()

spark = SparkSession.builder.getOrCreate()

df = spark.read.csv('../data/ml-25m/ratings.csv', header=True, inferSchema=True)

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


# Identify the movie with the greatest decline
df_falling = df_diff.orderBy('diff', ascending=True).first()
df_falling_filtered = df_yearly.filter(df_yearly['movieId'] == df_falling['movieId'])


# Find the corresponding movies' name
df_title = spark.read.csv('../data/ml-25m/movies.csv', header=True, inferSchema=True)
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
plt.show()

#Draw a line chart of the movie with the largest drop
plt.figure(figsize=(10, 6))
plt.plot(df_falling_filtered['Year'], df_falling_filtered['cumulative_avg_rating'], marker='o', color='g')
for i, row in df_falling_filtered.iterrows():
    plt.text(row['Year'], row['cumulative_avg_rating'], f"{row['cumulative_avg_rating']:.2f}")
plt.title(f'Rating trend of \"{falling_title}\"')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.xticks(df_falling_filtered['Year'], rotation=90)
plt.show()

#1_3_2_最受欢迎题材

import seaborn as sns

# Create SparkSession
spark = SparkSession.builder.getOrCreate()

df_rating = spark.read.csv('../data/ml-25m/ratings.csv', header=True, inferSchema=True)
df_movie = spark.read.csv('../data/ml-25m/movies.csv', header=True, inferSchema=True)

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

# Convert to Pandas DataFrame
df_rating_count_pandas = df_rating_count.toPandas()

# Draw the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='pop', y='genre', data=df_rating_count_pandas, orient='h')
plt.title('Popularity of Genres')
plt.xlabel('Popularity')
plt.ylabel('Genre')
plt.show()

#1_3_3_评分差距最小的电影（按题材分类）

spark = SparkSession.builder.getOrCreate()

# Read the csv file and create a DataFrame
df = spark.read.csv('../data/ml-25m/ratings.csv', header=True, inferSchema=True)

# Read the movie titles with genres
df_movies = spark.read.csv('../data/ml-25m/movies.csv', header=True, inferSchema=True)

# Join the ratings with movie titles on movieId
df_with_genres = df.join(df_movies, 'movieId')

# Calculate the best and worst rating for each movie genre
df_best = df_with_genres.groupBy('genres').agg(max('rating').alias('best_rating'))
df_worst = df_with_genres.groupBy('genres').agg(min('rating').alias('worst_rating'))

# Calculate the difference between best and worst ratings for each genre
df_diff = df_best.join(df_worst, 'genres')
df_diff = df_diff.withColumn('rating_diff', abs(col('best_rating') - col('worst_rating')))

# Filter out genres with a rating difference of 0 and find the bottom 50 genres with the minimum rating difference
df_filtered_diff = df_diff.filter(df_diff['rating_diff'] > 0)
df_bottom_50_diff = df_filtered_diff.orderBy('rating_diff').limit(50)

# Find the genre with the minimum difference
df_min_diff = df_filtered_diff.orderBy('rating_diff').limit(1)

# Output the result
df_bottom_50_diff.show()

# Output the genre with the minimum difference
df_min_diff.show()

# Convert the result to a Pandas DataFrame
df_bottom_50_diff_pd = df_bottom_50_diff.toPandas()
df_bottom_50_diff_pd.set_index('genres', inplace=True)

# Draw a bar chart of the best and worst ratings for the bottom 50 genres with the minimum difference
plt.figure(figsize=(10, 6))
df_bottom_50_diff_pd.plot.bar()
plt.xticks(rotation=90)
plt.title('Best and Worst Ratings for Bottom 50 (by Genres)')
plt.xlabel('Genre')
plt.ylabel('Rating Value')
plt.legend()
plt.show()

#1_3_4_50部评分差距最小的电影（横向对比评分高低）

spark = SparkSession.builder.getOrCreate()

# Read the csv file and create a DataFrame
df = spark.read.csv('../data/ml-25m/ratings.csv', header=True, inferSchema=True)

# Calculate the best and worst rating for each movie
df_best = df.groupBy('movieId').agg(max('rating').alias('best_rating'))
df_worst = df.groupBy('movieId').agg(min('rating').alias('worst_rating'))

# Calculate the difference between best and worst ratings
df_diff = df_best.join(df_worst, 'movieId')
df_diff = df_diff.withColumn('rating_diff', abs(col('best_rating') - col('worst_rating')))

# Filter out movies with a rating difference of 0 and find the bottom 50 movies with the minimum rating difference
df_filtered_diff = df_diff.filter(df_diff['rating_diff'] > 0)
df_bottom_50_diff = df_filtered_diff.orderBy('rating_diff').limit(50)

# Find the movie with the minimum difference
df_min_diff = df_filtered_diff.orderBy('rating_diff').limit(1)

# Read the movie titles
df_title = spark.read.csv('../data/ml-25m/movies.csv', header=True, inferSchema=True)

# Join with movie titles to get the names of the movies with the bottom 50 rating differences
df_bottom_50_diff_with_titles = df_bottom_50_diff.join(df_title, 'movieId')

# Output the result
df_bottom_50_diff_with_titles.show(50)

# Output the movie with the minimum difference
df_min_diff.show()

# Convert the result to a Pandas DataFrame
df_bottom_50_diff_pd = df_bottom_50_diff_with_titles.toPandas()
df_bottom_50_diff_pd.set_index('title', inplace=True)

# Draw a line chart of the best and worst ratings for the bottom 50 movies with the minimum difference
plt.figure(figsize=(10, 6))
df_bottom_50_diff_pd.plot.bar()
plt.xticks(rotation=90)
plt.title('Best and Worst Ratings for Bottom 50 Movies')
plt.xlabel('Movie')
plt.ylabel('Rating Value')
plt.legend()
plt.show()

#1_3_5 按照评分极差分桶展示各个桶的电影数量

spark = SparkSession.builder.getOrCreate()

# Read the csv file and create a DataFrame
df = spark.read.csv('../data/ml-25m/ratings.csv', header=True, inferSchema=True)

# Calculate the best and worst rating for each movie
df_best = df.groupBy('movieId').agg(max('rating').alias('best_rating'))
df_worst = df.groupBy('movieId').agg(min('rating').alias('worst_rating'))

# Calculate the difference between best and worst ratings
df_diff = df_best.join(df_worst, 'movieId')
df_diff = df_diff.withColumn('rating_diff', abs(col('best_rating') - col('worst_rating')))

# Define the rating difference bins
bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

# Categorize the rating differences into bins
df_diff = df_diff.withColumn('rating_diff_bin', when(col('rating_diff') == 0, '0')
                                           .when(col('rating_diff') <= 0.5, '0.5')
                                           .when(col('rating_diff') <= 1, '1')
                                           .when(col('rating_diff') <= 1.5, '1.5')
                                           .when(col('rating_diff') <= 2, '2')
                                           .when(col('rating_diff') <= 2.5, '2.5')
                                           .when(col('rating_diff') <= 3, '3')
                                           .when(col('rating_diff') <= 3.5, '3.5')
                                           .when(col('rating_diff') <= 4, '4')
                                           .when(col('rating_diff') <= 4.5, '4.5')
                                           .otherwise('5'))

# Count the number of movies in each rating difference bin
rating_diff_counts = df_diff.groupBy('rating_diff_bin').count().orderBy('rating_diff_bin')

# Output the result
rating_diff_counts.show()

# Convert the result to a Pandas DataFrame
rating_diff_counts_pd = rating_diff_counts.toPandas()

# Draw a bar chart of the movie count for different rating differences using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='rating_diff_bin', y='count', data=rating_diff_counts_pd, palette='viridis')
plt.title('Movie Count for Different Rating Differences')
plt.xlabel('Rating Difference')
plt.ylabel('Movie Count')
plt.show()

#1_3_6_口味最符合大众和最不符合大众口味的评分者（保留）

#1_3_7_Top 20 Tags by Average Relevance

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder.appName("Movie Analysis").getOrCreate()

# 加载数据集
genome_tags_df = spark.read.csv("../data/ml-25m/genome-tags.csv", header=True, inferSchema=True)
genome_scores_df = spark.read.csv("../data/ml-25m/genome-scores.csv", header=True, inferSchema=True)

# 求出每个标签的平均相关性
avg_tag_relevance = genome_scores_df.groupBy("tagId").agg({"relevance": "mean"})

# 加入标签名称
avg_tag_relevance = avg_tag_relevance.join(genome_tags_df, "tagId").select("tag", "avg(relevance)")

# 转换为 Pandas DataFrame 进行可视化
avg_tag_relevance_pd = avg_tag_relevance.toPandas()

# 可视化
plt.figure(figsize=(12, 8))
sns.barplot(x="avg(relevance)", y="tag", data=avg_tag_relevance_pd.sort_values("avg(relevance)", ascending=False).head(20))
plt.title("Top 20 Tags by Average Relevance")
plt.xlabel("Average Relevance")
plt.ylabel("Tag")
plt.show()

#1_3_8_Top 20 Most Popular Tags 

tags_df = spark.read.csv("../data/ml-25m/tags.csv", header=True, inferSchema=True)

# 计算最常用的标签
top_tags = tags_df.groupBy("tag").count().orderBy("count", ascending=False)

# 转换为 Pandas DataFrame 进行可视化
top_tags_pd = top_tags.toPandas()

# 可视化
plt.figure(figsize=(12, 8))
sns.barplot(x="count", y="tag", data=top_tags_pd.head(20))
plt.title("Top 20 Most Popular Tags")
plt.xlabel("Count")
plt.ylabel("Tag")
plt.show()

#1_3_9 每个年代最受欢迎Top5电影题材

spark = SparkSession.builder.getOrCreate()

# Read the csv files and create DataFrames
df_movies = spark.read.csv('../data/ml-25m/movies.csv', header=True, inferSchema=True)

# Extract the year from title
df_movies = df_movies.withColumn('year', regexp_extract(col('title'), '\((\d{4})\)', 1))

# Filter the data for the desired time range (1930-2022)
df_movies = df_movies.filter((df_movies['year'] >= 1930) & (df_movies['year'] <= 2022))

# Define a function to categorize the year into decades
def categorize_decade(year):
    if year:
        decade = str(int(year) // 10 * 10) + "s"
        return decade
    else:
        return None

categorize_decade_udf = udf(categorize_decade)

# Apply the function to categorize the year into decades
df_movies = df_movies.withColumn('decade', categorize_decade_udf(col('year')))

# Split the genres and explode the array to separate rows
df_movies = df_movies.withColumn('genre', explode(split(df_movies['genres'], '[|]')))

# Group by decade and genre, and count the occurrences
genre_counts = df_movies.filter(df_movies['genre'] != '(no genre listed)').groupBy('decade', 'genre').count()

# For each decade, find the top 5 most common genres
window = Window.partitionBy('decade').orderBy(desc('count'))
top_genre = genre_counts.withColumn('rank', dense_rank().over(window)).filter(col('rank') <= 5).select('decade', 'genre', 'count')

# Output the result
top_genre.show()

# Convert the result to a Pandas DataFrame
top_genre_pd = top_genre.toPandas()

# Pivot the DataFrame
top_genre_pd = top_genre_pd.pivot(index='decade', columns='genre', values='count').fillna(0)

# Remove the unwanted columns
unwanted_genres = ['(no genres listed)', 'Other']
for genre in unwanted_genres:
    if genre in top_genre_pd.columns:
        top_genre_pd = top_genre_pd.drop(columns=[genre])

# Draw a stacked bar chart of the top 5 most common genres for each decade using Seaborn
plt.figure(figsize=(12, 8))
top_genre_pd = top_genre_pd.div(top_genre_pd.sum(axis=1), axis=0)
ax = top_genre_pd.plot(kind='bar', stacked=True, cmap='viridis')
plt.title('Top 5 Most Common Genres for Each Decade')
plt.xlabel('Decade')
plt.ylabel('Movie Count')
plt.xticks(rotation=45)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Initialize a variable to store the cumulative height
cumulative_height = [0] * len(top_genre_pd)

for i, genre in enumerate(top_genre_pd.columns):
    heights = top_genre_pd[genre]
    for j, height in enumerate(heights):
        x = j
        y = cumulative_height[j] + height / 2
        percentage = height * 100  # Calculate the percentage
        ax.text(x, y, f'{percentage:.0f}%', ha='center', va='center', fontsize=8)  # Add the percentage as text
        cumulative_height[j] += height

plt.show()

#1_3_10 Genre Group Distribution (Sorted by Total Count)

spark = SparkSession.builder.getOrCreate()

df_rating = spark.read.csv('../data/ml-25m/ratings.csv', header=True, inferSchema=True)
df_movie = spark.read.csv('../data/ml-25m/movies.csv', header=True, inferSchema=True)
df_links = spark.read.csv('../data/ml-25m/links.csv', header=True, inferSchema=True)
df_gscores = spark.read.csv('../data/ml-25m/genome-scores.csv', header=True, inferSchema=True)
df_tags = spark.read.csv('../data/ml-25m/genome-tags.csv', header=True, inferSchema=True)

# Show the head of each DataFrame
print("df_rating:")
df_rating.show(3)

print("df_movie:")
df_movie.show(3)

print("df_links:")
df_links.show(3)

print("df_gscores:")
df_gscores.show(3)

print("df_tags:")
df_tags.show(3)

# Count the number of unique genres
unique_genres = df_movie.select(explode(split('genres', '\|')).alias('genre')).distinct()
num_unique_genres = unique_genres.count()
print(f"Number of unique genres: {num_unique_genres}")

# Count the occurrences of each genre
df_genres = df_movie.withColumn('genre', explode(split('genres', '\|')))
genre_counts = df_genres.groupBy('genre').count()
genre_counts.show()

# filter out 
# Assuming your DataFrame is df_movie and it has a 'genres' column
total_movies_before_explode = df_movie.select(count("movieId")).collect()[0][0]

df_genres = df_movie.withColumn('genre', explode(split('genres', '\|')))

# Filter out 'IMAX' and '(no genres listed)'

df_genres_filtered = df_genres.filter((col('genre') != 'IMAX') & (col('genre') != '(no genres listed)'))

from pyspark.sql.functions import col, when
from pyspark.ml.feature import Bucketizer
from pyspark.sql import functions as F

# Assuming your DataFrame is genre_counts
df_grouped_genres = genre_counts.withColumn(
    'genre_group',
    when(col('genre').isin(['Crime', 'Mystery']), 'Crime/Mystery') \
    .when(col('genre').isin(['Romance', 'Musical', 'Drama']), 'Romance/Musical/Drama') \
    .when(col('genre').isin(['Adventure', 'Fantasy', 'Animation']), 'Adventure/Fantasy/Animation') \
    .when(col('genre').isin(['Documentary', 'War']), 'Documentary/War') \
    .when(col('genre').isin(['Horror', 'Thriller']), 'Horror/Thriller') \
    .when(col('genre').isin(['Comedy', 'Children']), 'Comedy/Children') \
    .when(col('genre').isin(['Action', 'Sci-Fi']), 'Action/Sci-Fi') \
    .otherwise('Other')
)

# Group by the new genre groups
grouped_genre_counts = df_grouped_genres.groupBy('genre_group').sum('count').withColumnRenamed('sum(count)', 'total_count')

# Show the result
grouped_genre_counts.show(truncate=False)

# Convert PySpark DataFrame to pandas DataFrame
grouped_genre_counts_pd = grouped_genre_counts.toPandas()

# Sort the DataFrame by "total_count" in ascending order
grouped_genre_counts_pd = grouped_genre_counts_pd.sort_values(by="total_count", ascending=True)

# Set up Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Use the "husl" palette for a rainbow-like color scheme
palette = sns.color_palette("husl", n_colors=len(grouped_genre_counts_pd))

# Create a bar plot
ax = sns.barplot(x="total_count", y="genre_group", data=grouped_genre_counts_pd, palette=palette)

# Add total number of movies annotation
plt.annotate(f'Total Movies\n{total_movies_before_explode}', (0.5, -0.15), xycoords="axes fraction", ha='center', va='center', fontsize=12, color='black')

plt.xlabel('Total Count')
plt.ylabel('Genre Group')
plt.title('Genre Group Distribution (Sorted by Total Count)')
plt.show()

#1_3_11 Genre Group Distribution

# Assuming your DataFrame is df_movie
total_movies_before_explode = df_movie.select(count("movieId")).collect()[0][0]

# Assuming your DataFrame is df_movie and it has a 'genres' column
df_genres = df_movie.withColumn('genre', explode(split('genres', '\|')))

# Filter out 'IMAX' and '(no genres listed)'
df_genres_filtered = df_genres.filter((col('genre') != 'IMAX') & (col('genre') != '(no genres listed)'))


# Convert PySpark DataFrame to pandas DataFrame
grouped_genre_counts_pd = grouped_genre_counts.toPandas()

# Set up Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))

# Use the "husl" palette for a rainbow-like color scheme
colors = sns.color_palette("husl", n_colors=len(grouped_genre_counts_pd))

# Create a pie chart with labels and percentages
plt.pie(grouped_genre_counts_pd['total_count'], labels=grouped_genre_counts_pd['genre_group'], autopct='%1.1f%%', colors=colors)

# Draw a circle at the center of the pie to make it look like a donut chart
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Add text annotation for movie counts using total_movies_before_explode
plt.annotate(f'Total Movies\n{total_movies_before_explode}', (0, 0), fontsize=12, ha='center', va='center', color='black')

plt.title('Genre Group Distribution')

plt.show()

#1_3_12 Distribution of Ratings for Each Genre Group

spark = SparkSession.builder.getOrCreate()

# Assuming you have the DataFrames df_rating and df_movie
# Match the movie ratings with the movie IDs
df_movie_rating = df_rating.select("movieId", "rating").join(df_movie, on='movieId')

# Filter out rows where 'rating' is not a numerical value
df_movie_rating = df_movie_rating.filter(df_movie_rating['rating'].cast('float').isNotNull())

# Explode by genres
df_genres = df_movie_rating.withColumn('genre', explode(split('genres', '\|')))

# Filter out 'IMAX' and '(no genres listed)'
df_genres_filtered = df_genres.filter((col('genre') != 'IMAX') & (col('genre') != '(no genres listed)'))

# Perform genre grouping based on specified rules
df_grouped_genres = df_genres_filtered.withColumn(
    'genre_group',
    when(col('genre').isin(['Crime', 'Mystery']), 'Crime/Mystery') \
    .when(col('genre').isin(['Romance', 'Musical', 'Drama']), 'Romance/Musical/Drama') \
    .when(col('genre').isin(['Adventure', 'Fantasy', 'Animation']), 'Adventure/Fantasy/Animation') \
    .when(col('genre').isin(['Documentary', 'War']), 'Documentary/War') \
    .when(col('genre').isin(['Horror', 'Thriller']), 'Horror/Thriller') \
    .when(col('genre').isin(['Comedy', 'Children']), 'Comedy/Children') \
    .when(col('genre').isin(['Action', 'Sci-Fi']), 'Action/Sci-Fi') \
    .otherwise('Other')
)


# Drop unnecessary columns
df_grouped_genres = df_grouped_genres.drop('title', 'genres', 'genre')

# Create a Bucketizer to bin the ratings with 0.5 intervals from 0 to 5
# Create a Bucketizer to bin the ratings with 0.5 intervals from 0 to 5
bucketizer = Bucketizer(splits=[-0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25],
                        inputCol="rating", outputCol="rating_bin")

# Apply the Bucketizer to the DataFrame
df_binned_ratings = bucketizer.setHandleInvalid("keep").transform(df_grouped_genres)

# Group by genre_group and rating_bin, and count occurrences
df_rating_distribution = df_binned_ratings.groupBy("genre_group", "rating_bin").agg(count("*").alias("count"))

# Calculate the total count for each genre group
window_spec = Window.partitionBy("genre_group")
df_rating_distribution = df_rating_distribution.withColumn("total_count", F.sum("count").over(window_spec))

# Calculate the percentage for each bin within each genre group
df_rating_distribution = df_rating_distribution.withColumn("percentage", (col("count") / col("total_count")) * 100)

# Convert PySpark DataFrame to Pandas DataFrame
df_rating_distribution_pd = df_rating_distribution.toPandas()

# Set up Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Create a bar plot for each genre group
ax = sns.barplot(x="rating_bin", y="percentage", hue="genre_group", data=df_rating_distribution_pd, palette="husl")

# Customize x-axis ticks and labels
midpoints = [i * 0.5 + 0.5 for i in range(10)]  # Midpoints of each bin
ax.set_xticks(range(10))  # 10 ticks for 10 bins
ax.set_xticklabels([f'{mid:.1f}' for mid in midpoints], rotation=45, ha="right")

plt.xlabel('Rating')
plt.ylabel('Percentage')
plt.title('Distribution of Ratings for Each Genre Group')
plt.show()

spark.stop()
end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "seconds")

