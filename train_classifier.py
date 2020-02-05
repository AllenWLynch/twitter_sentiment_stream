
#%%
import findspark
findspark.init()

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql import *
from pyspark.sql import types
from pyspark.sql import functions
from pyspark import ml
from nltk.tokenize import TweetTokenizer

spark = SparkSession.builder.appName("Train SVM").getOrCreate()

#%%
import importlib

#%%
import pipeline_utils
pipeline_utils = importlib.reload(pipeline_utils)
from nltk.corpus import stopwords

#%%
if __name__ == "__main__":

    tweets = spark.read.load('data/training.csv', format = 'csv', sep = ',', schema = pipeline_utils.TWEET_DB_SCHEMA)
    tweets = tweets.withColumn('label', functions.when(tweets['label']==0, 0).otherwise(1))
    tweets.createOrReplaceTempView('tweets')
#%%
    TRAIN_SIZE = 60000
    TEST_SIZE = 5000
    train = tweets.sample(False, TRAIN_SIZE/tweets.count())
    test = tweets.sample(False, TEST_SIZE/tweets.count())

#%%
    def getROC(model, test_set):

        predictions = model.transform(test_set)
        evaluator = ml.evaluation.BinaryClassificationEvaluator(labelCol='label', rawPredictionCol ='rawPrediction')

        auc = evaluator.evaluate(predictions)

        return auc

    def search_hyperparams(model, regressor, training_set):
        
        paramGrid = ml.tuning.ParamGridBuilder().addGrid(regressor.regParam, [0.1,0.01,0.001]).addGrid(regressor.elasticNetParam,[0.0, 0.5, 1.0]).build()
        #evaluates with area under ROC curve
        evaluator = ml.evaluation.BinaryClassificationEvaluator()
        tvs = ml.tuning.TrainValidationSplit(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.9)
        bestModel = tvs.fit(training_set).bestModel

        return bestModel

    all_stopwords = set(ml.feature.StopWordsRemover.loadDefaultStopWords('english')).union(set(stopwords.words('english')))
#%%
#1-grams LR model
    def define_unigram_model():
        VOCAB_SIZE = 20000
        MINDF = 3
        TRAINING_ITERS = 150

        tokenizer = pipeline_utils.TweetTokenizer(inputCol='text',outputCol='words')
        stopword_remover = ml.feature.StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='filtered', stopWords=list(all_stopwords))
        stemmer = pipeline_utils.Stemmer(inputCol=stopword_remover.getOutputCol(), outputCol='cleaned_words')
        counter = ml.feature.CountVectorizer(inputCol=stemmer.getOutputCol(), outputCol='counts',vocabSize=VOCAB_SIZE, minDF=MINDF)
        normalizer = ml.feature.Normalizer(p = 1.0, inputCol=counter.getOutputCol(), outputCol='tf_normalized')
        df_normalize = ml.feature.IDF(inputCol=normalizer.getOutputCol(), outputCol='features')
        #regresser = ml.classification.LogisticRegression(maxIter= TRAINING_ITERS, regParam=0.01,elasticNetParam=0.5, 
        #    featuresCol='features', labelCol= 'label')
        regresser = ml.classification.MultilayerPerceptronClassifier(maxIter=TRAINING_ITERS, layers = [3,2,1], blockSize=64,seed=1234)
        pipeline = ml.Pipeline(stages=[tokenizer, stopword_remover, stemmer, counter, normalizer, df_normalize, regresser])

        return pipeline, regresser

    unigram_model, regresser = define_unigram_model()
#%%

    best_unigram_model = search_hyperparams(unigram_model, regresser, train)

    getROC(best_unigram_model, test)

    best_unigram_model.save('best_unigram_model')

#%%
    mlp_model, _ = define_unigram_model()

    trained_mlp = mlp_model.fit(train)

    getROC(trained_mlp, test)

#%%
#bigrams LR model
    def define_bigram_model():
        VOCAB_SIZE = 20000
        MINDF = 3
        TRAINING_ITERS = 150

        tokenizer = pipeline_utils.TweetTokenizer(inputCol='text',outputCol='words')
        #stopword_remover = ml.feature.StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='filtered', stopWords=list(all_stopwords))
        stemmer = pipeline_utils.Stemmer(inputCol=tokenizer.getOutputCol(), outputCol='cleaned_words')
        ngrammer = ml.feature.NGram(n=3, inputCol=stemmer.getOutputCol(), outputCol='ngrams')
        counter = ml.feature.CountVectorizer(inputCol=ngrammer.getOutputCol(), outputCol='counts',vocabSize=VOCAB_SIZE, minDF=MINDF)
        normalizer = ml.feature.Normalizer(p = 1.0, inputCol=counter.getOutputCol(), outputCol='tf_normalized')
        df_normalize = ml.feature.IDF(inputCol=normalizer.getOutputCol(), outputCol='features')
        regresser = ml.classification.LogisticRegression(maxIter= TRAINING_ITERS, 
            featuresCol='features', labelCol= 'label')
        pipeline = ml.Pipeline(stages=[tokenizer, stemmer, ngrammer, counter, normalizer, df_normalize, regresser])

        return pipeline, regresser

    bigram_model, bigram_regresser = define_bigram_model()

#%%

    best_bigram_model = search_hyperparams(bigram_model, bigram_regresser, train)

    getROC(best_bigram_model, test)

#%%

    def TFIDF_pipeline(prefix, inputCol, vocab_size, min_df = 3):
        counter = ml.feature.CountVectorizer(inputCol=inputCol, outputCol= prefix + '_counts',vocabSize=vocab_size, minDF=min_df)
        normalizer = ml.feature.Normalizer(p = 1.0, inputCol=counter.getOutputCol(), outputCol=prefix + '_tf_normalized')
        df_normalize = ml.feature.IDF(inputCol=normalizer.getOutputCol(), outputCol=prefix + '_features')
        return ml.Pipeline(stages=[counter, normalizer, df_normalize])

    def define_combo_model():
        VOCAB_SIZE = 10000
        MINDF = 3
        TRAINING_ITERS = 150

        tokenizer = pipeline_utils.TweetTokenizer(inputCol='text',outputCol='words')
        stemmer = pipeline_utils.Stemmer(inputCol=tokenizer.getOutputCol(), outputCol='cleaned_words')

        stopword_remover = ml.feature.StopWordsRemover(inputCol=stemmer.getOutputCol(), outputCol='stopwords_removed', stopWords=list(all_stopwords))
        unigram_ifidf = TFIDF_pipeline('unigram',stopword_remover.getOutputCol(), 15000)
        
        ngrammer = ml.feature.NGram(n=3, inputCol=stemmer.getOutputCol(), outputCol='trigrams')
        ngram_ifidf = TFIDF_pipeline('trigram', ngrammer.getOutputCol(), 5000)

        assembler = ml.feature.VectorAssembler(inputCols=['unigram_features', 'trigram_features'], outputCol='features')
        
        regresser = ml.classification.LogisticRegression(maxIter= TRAINING_ITERS, 
            featuresCol='features', labelCol= 'label')
        pipeline = ml.Pipeline(stages=[tokenizer, stemmer, stopword_remover, unigram_ifidf, ngrammer, ngram_ifidf, assembler, regresser])

        return pipeline, regresser

    mixed_pipeline = define_combo_model()

#%%

    best_mixed_model = search_hyperparams(*mixed_pipeline, train)

    getROC(best_mixed_model, test)



# %%

    def defineNB_model():
        VOCAB_SIZE = 20000
        MINDF = 3
        TRAINING_ITERS = 150

        tokenizer = pipeline_utils.TweetTokenizer(inputCol='text',outputCol='words')
        stopword_remover = ml.feature.StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='filtered', stopWords=list(all_stopwords))
        stemmer = pipeline_utils.Stemmer(inputCol=stopword_remover.getOutputCol(), outputCol='cleaned_words')
        counter = ml.feature.CountVectorizer(inputCol=stemmer.getOutputCol(), outputCol='features',vocabSize=VOCAB_SIZE, minDF=MINDF)
        
        classifier = ml.classification.NaiveBayes(smoothing = 1.0, modelType='multinomial')
        pipeline = ml.Pipeline(stages=[tokenizer, stopword_remover, stemmer, counter, classifier])

        return pipeline, classifier

    nb_pipeline, nb_classifier = defineNB_model()

    paramGrid = ml.tuning.ParamGridBuilder().addGrid(nb_classifier.smoothing, [0, 0.5, 1., 10., 100.]).build()
    #evaluates with area under ROC curve
    evaluator = ml.evaluation.BinaryClassificationEvaluator()
    tvs = ml.tuning.TrainValidationSplit(estimator=nb_pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.9)
    bestModel = tvs.fit(train).bestModel

    getROC(bestModel, test)


# %%
