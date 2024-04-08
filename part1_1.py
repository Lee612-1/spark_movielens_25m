import time
from pyspark import SparkContext

start_time = time.time()

# Create SparkContext
sc = SparkContext("local", "RatingCount")

# Read the rating file
lines = sc.textFile("../data/ml-25m/ratings.csv")

# Skip the header line
header = lines.first()
lines = lines.filter(lambda line: line != header)

# Parse the lines and create an RDD
ratings = lines.map(lambda line: line.split(","))

# Count the number of ratings
count = ratings.count()

print("Number of ratings:", count)

# Stop SparkContext
sc.stop()

end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "seconds")
