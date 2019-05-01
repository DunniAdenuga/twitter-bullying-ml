
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import load_data
import train_ngram_model
import tune_ngram_model

data = load_data.load_tweets()
# acc, loss = train_ngram_model.train_ngram_model(data)
train_ngram_model.train_ngram_model(data)
# print("acc: " + str(acc))
# print("loss: " + str(loss))
#
tune_ngram_model.tune_ngram_model(data)

# try with test data doubled
