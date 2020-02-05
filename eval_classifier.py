
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
import sklearn
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import pandas as pd

#%%
if __name__ == "__main__":

    tweets = spark.read.load('./data/test.csv', format = 'csv', sep = ',', schema = pipeline_utils.TWEET_DB_SCHEMA)
    if False:
        #BIN ALL NUETRAL + NEGATIVE
        tweets = tweets.withColumn('label', functions.when(tweets['label']==4, 1).otherwise(0))
    elif False:
        #BIN ALL NEURAL + POSITIVE
        tweets = tweets.withColumn('label', functions.when(tweets['label']==0, 0).otherwise(1))
    elif True:
        tweets = tweets.withColumn('label', tweets['label'] / 2)

    tweets.select('label').distinct().show()

#%%
    model = ml.PipelineModel.load('best_unigram_model')
    
#%%
    predictions = model.transform(tweets).select('prediction','label','probability').toPandas()
    predictions['probability'] = predictions['probability'].apply(lambda x : x[1])
#%%
    NEG_THRESHOLD = 0.33
    POS_THRESHOLD = 0.66
    def get_label_from_prob(prob):
        if prob <= NEG_THRESHOLD:
            return 0
        elif prob >= POS_THRESHOLD:
            return 2
        else:
            return 1
    predictions['prediction'] = predictions['probability'].apply(get_label_from_prob)
#%%

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(predictions['label'], predictions['probability'])

# %%
    thresholds[0] = 1
    cmap = cm.jet(thresholds)
    fig = plt.figure(figsize=(9,6.5))
    plt.scatter(fpr, tpr, c= cmap)
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.title('Negative sentiment ROC Curve, Logistic Regression')

    scalarmap = cm.ScalarMappable(norm = mcolors.Normalize(vmin=0,vmax=1) , cmap = cm.jet)
    scalarmap.set_array(thresholds)
    plt.colorbar(scalarmap).set_label('Threshold')
    plt.savefig('neg_roc_curve.png')

# %%

    confusion_matrix = sklearn.metrics.confusion_matrix(predictions['label'], predictions['prediction'])
    #conf_norm = sklearn.metrics.confusion_matrix(np_predictions[:,1], np_predictions[:,0], labels=[0,1], normalize='all')
    confusion_matrix

#%%
    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    plt.table(cellText = confusion_matrix.astype(str), rowLabels = ['Neg','Pos'], colLabels = ['Neg','Pos'], loc ='center')
    fig.tight_layout()
    plt.xlabel('Label')
    plt.ylabel('Prediction')

    plt.show()


# %%

    #nuetral analysis
    np_predictions

# %%
