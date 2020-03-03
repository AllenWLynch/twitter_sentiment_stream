

#%%
import findspark
findspark.init()

import pyspark

spark = pyspark.sql.SparkSession.builder.appName("Tweet D-stream")\
    .config("spark.executor.memory","10g")\
    .config("spark.driver.memory","10g")\
    .getOrCreate()

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark import ml
import os
import datetime
# %%

tweet_schema = T.StructType().add("created_at",T.StringType()).add("text",T.StringType()).add("hashtags",T.ArrayType(T.StringType()))\
        .add("place_name", T.StringType()).add("lon", T.FloatType()).add("lat",T.FloatType())

POS_THRESHOLD = 0.66
NEG_THRESHOLD = 0.33
#%%
MODEL_NAME = 'best_unigram_model'

pos_model = ml.PipelineModel.load(MODEL_NAME)
neg_model = ml.PipelineModel.load(MODEL_NAME)

pos_model.stages[-1].setThreshold(POS_THRESHOLD)
neg_model.stages[-1].setThreshold(NEG_THRESHOLD)


#%%
HOST = 'localhost'
PORT = 9009
BASE_DIR = './assets/data'
#TWEETS_OUTPUT = './streamed_data/tweets'
#BATCH_DIAGNOSTIC_OUTPUT = './streamed_data/batches'
SESSION_ID = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
DATA_DIR = os.path.join(BASE_DIR, SESSION_ID)

TWEETS_OUTPUT = os.path.join(DATA_DIR, 'tweets')
BATCH_DIAGNOSTIC_OUTPUT = os.path.join(DATA_DIR, 'batches')

for filepath in (DATA_DIR, TWEETS_OUTPUT, BATCH_DIAGNOSTIC_OUTPUT):
    os.mkdir(filepath)

#%%
def save_function(df, batch_id):
    df.persist()
    df.write.mode("append").format("parquet").save(TWEETS_OUTPUT)
    agg_df = df.agg(F.min("created_at_unix").alias("created_at_unix"), F.count("text").alias("count"))
    agg_df.persist()
    agg_df = agg_df.withColumn("timestamp", F.current_timestamp())
    agg_df = agg_df.withColumn("latency", (F.unix_timestamp("timestamp") - F.col("created_at_unix").alias("latency")))
    agg_df = agg_df.withColumn("batch_id", F.lit(batch_id))
    agg_df.write.mode("append").format("parquet").save(BATCH_DIAGNOSTIC_OUTPUT)
    agg_df.unpersist()
    df.unpersist()

#%%
streamed_tweets = spark.readStream\
        .format("socket")\
        .option("host", HOST)\
        .option("port", PORT)\
        .load()

#%%
#receive -> convert to columns -> convert dates -> save

#1. convert to columns
streamed_tweets = streamed_tweets.withColumn('json', F.from_json("value", tweet_schema)).select("json.*")

#2. apply model
streamed_tweets = pos_model.transform(streamed_tweets)\
    .select("created_at","text","hashtags","place_name","lon","lat","features", F.col("prediction").alias("positive_label"))

streamed_tweets = neg_model.stages[-1].transform(streamed_tweets)\
    .select("created_at","text","hashtags","place_name","lon","lat",(F.col("prediction") + F.col("positive_label")).alias("label"))

#4. reformat dates and calc latency
streamed_tweets = streamed_tweets.withColumn("created_at_unix", F.unix_timestamp("created_at", format = "EEE MMM dd HH:mm:ss Z yyyy")) #unixtime
streamed_tweets = streamed_tweets.withColumn("created_at_timestamp", F.from_unixtime("created_at_unix"))

query = streamed_tweets \
    .writeStream \
    .foreachBatch(save_function)\
    .start()

query.awaitTermination()

#%%