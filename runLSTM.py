import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb

import load_data
import vectorize_data

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=2)

print("x-train")
print(X_train)
print("y-train")
print(y_train)
# fix random seed for reproducibility
# numpy.random.seed(7)
#
# data = load_data.load_tweets(lstm_flag=True)
# (train_texts, train_labels), (val_texts, val_labels) = data
#
# max_review_length = 50
#
# # x_train, x_val = vectorize_data.ngram_vectorize(
# #         train_texts, train_labels, val_texts)
#
# X_train = sequence.pad_sequences(train_texts, maxlen=max_review_length)
# X_test = sequence.pad_sequences(val_texts, maxlen=max_review_length)

# create the model
# embedding_vecor_length = 32
# model = Sequential()
# model.add(Embedding(X_train.length + X_test.length, embedding_vecor_length, input_length=max_review_length))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(X_train, train_labels, validation_data=(X_test, val_labels), epochs=3, batch_size=64)
#
# scores = model.evaluate(X_test, val_labels, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
