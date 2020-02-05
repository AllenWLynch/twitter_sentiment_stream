
#%%
import nltk
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol,Param,Params,TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable  
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import string
from pyspark.sql import types
from pyspark.sql import functions

#%%
#nltk.download('all')

field_names = ['label','id','date','query','user','text']
field_types = [types.IntegerType(), types.IntegerType(), types.DateType(), types.StringType(), types.StringType(), types.StringType()]

TWEET_DB_SCHEMA = types.StructType([
    types.StructField(field_name, field_type, True) for field_name, field_type in zip(field_names, field_types)
])

#%%
def load_tweet_db(spark_session, filepath, filter=True):

    tweets = spark_session.read.load(filepath, format = 'csv', sep = ',', schema = TWEET_DB_SCHEMA)
    if filter:
        tweets = tweets.filter(tweets['label'] != 2)
    tweets = tweets.withColumn('label', functions.when(tweets['label']==0, 0).otherwise(1))
    tweets = tweets.withColumn('label', 1 - tweets['label'])
    
    return tweets

#%%
class TweetTokenizer(
        Transformer, HasInputCol, HasOutputCol,
        DefaultParamsReadable, DefaultParamsWritable):
    tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False)
    punctuation = list(string.punctuation)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # Required in Spark >= 3.0
    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    # Required in Spark >= 3.0
    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def _transform(self, dataset):
        
        def tokenize_tweet(tweet):
            return [token for token in self.tokenizer.tokenize(tweet) if not token in self.punctuation]

        output_type = ArrayType(StringType())

        out_col = self.getOutputCol()

        in_col = dataset[self.getInputCol()]

        return dataset.withColumn(out_col, udf(tokenize_tweet, output_type)(in_col))

#%%
class Stemmer(
        Transformer, HasInputCol, HasOutputCol, DefaultParamsWritable, DefaultParamsReadable):

    stemmer = nltk.stem.PorterStemmer()

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # Required in Spark >= 3.0
    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    # Required in Spark >= 3.0
    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def _transform(self, dataset):
        def stem_tweet(words):
            return [self.stemmer.stem(word) for word in words]
        output_type = ArrayType(StringType())
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(stem_tweet, output_type)(in_col))

class ThresholdClassifier(Transformer, HasInputCol, HasOutputCol, 
    DefaultParamsWritable, DefaultParamsReadable):

    def __init__(self):
        pass



#%%
if __name__ == "__main__":
    sentenceDataFrame = spark.createDataFrame([
    (0, 1.1, "I wanted to do some gaming because I'm a gamer. Games make me happy."),
    (0, 1.2, "I wish Java could use case classes"),
    (1, 1.3, "Logistic regression models are neat. Sometimes."),
    (2, 1.4, "I am testing the stemmitization model, right?")
    ], ["label", "x1", "sentence"])

    #%%
    
    testTokenizer = TweetTokenizer(inputCol="sentence", outputCol="words")
    stemmer = Stemmer(inputCol='words',outputCol='stems')

    #%%
    df = testTokenizer.transform(sentenceDataFrame)
    df = stemmer.transform(df)

    # %%
    df.show()
