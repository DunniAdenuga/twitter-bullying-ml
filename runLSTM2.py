
import load_data

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

# Vectorization parameters
# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 500

def sequence_vectorize(train_texts, val_texts):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length3 = len(max(x_train, key=len))
    if max_length3 > MAX_SEQUENCE_LENGTH:
        max_length3 = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length3)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length3)
    return x_train, x_val, tokenizer.word_index, max_length3

((train_texts, train_labels),
            (test_texts, test_labels), max_length2) = load_data.load_tweets(lstm_flag=True)
(train_texts2, y_train2), (val_texts2, y_test2) = load_data.load_tweets(lstm_flag=False)
X_train, X_test, word_index, max_length = sequence_vectorize(train_texts2, val_texts2)
print("max-len")
print(max_length)

def word_count(sentence_list):
    wc_dict = dict()
    for sentence in sentence_list:
        for word in sentence:
            if word in wc_dict:
                wc_dict[word] += 1
            else:
                wc_dict[word] = 1
    return wc_dict

# wc_dict = word_count(X_train)
wc_dict = word_count(train_texts)

print(len(wc_dict))

sentences = train_texts

# def build_word_to_index(wc_dict, vocab_size):
#     wc_list = []
#     for word in wc_dict:
#         wc_list.append((wc_dict[word], word))
#     wc_list.sort()
#     wc_list = wc_list[::-1]
#     word_to_index = dict()
#     word_to_index["<START>"] = 1
#     word_to_index["<UNK>"] = 2
#     for i in range(vocab_size):
#         word_to_index[wc_list[i][1]] = i + 3
#     return word_to_index
#
#
# word_to_index = build_word_to_index(wc_dict, 63)
# print(len(word_to_index))

# X_train_index = []
# for sentence in X_train
#     sentence_index = [1]
#     for word in sentence:
#         if word in word_to_index:
#             sentence_index.append(word_to_index[word])
#         else:
#             sentence_index.append(2)
#     X_train_index.append(sentence_index)

from random import shuffle

# def split_train_test(X, y):
#     X_pos = []
#     X_neg = []
#     y_pos = []
#     y_neg = []
#     for i in range(len(y)):
#         if y[i] == 0:
#             X_neg.append(X[i])
#             y_neg.append(y[i])
#         else:
#             X_pos.append(X[i])
#             y_pos.append(y[i])
#     pos_seq = [i for i in range(len(y_pos))]
#     neg_seq = [i for i in range(len(y_neg))]
#     shuffle(pos_seq)
#     shuffle(neg_seq)
#     X_train = []
#     y_train = []
#     X_test = []
#     y_test = []
#     for i in range(len(y_pos)):
#         if i < 0.8*len(y_pos):
#             X_train.append(X_pos[i])
#             y_train.append(y_pos[i])
#         else:
#             X_test.append(X_pos[i])
#             y_test.append(y_pos[i])
#     for i in range(len(y_neg)):
#         if i < 0.8*len(y_neg):
#             X_train.append(X_neg[i])
#             y_train.append(y_neg[i])
#         else:
#             X_test.append(X_neg[i])
#             y_test.append(y_neg[i])
#     return X_train, y_train, X_test, y_test
#
#
# X_train, y_train, X_test, y_test = split_train_test(X_train_index, y_train)

X_train_new = []
y_train_new = []
oversampling_multi = 7

for i in range(len(y_train2)):
    if y_train2[i] == 1:
        for j in range(oversampling_multi):
            y_train_new += [y_train2[i]]
            X_train_new += [X_train[i]]
    else:
        y_train_new += [y_train2[i]]
        X_train_new += [X_train[i]]


# randomize the sequence
def rand_seq(X_list, y_list):
    new_seq = [i for i in range(len(y_list))]
    shuffle(new_seq)
    X_ret = []
    y_ret = []
    for i in range(len(new_seq)):
        X_ret.append(X_list[new_seq[i]])
        y_ret.append(y_list[new_seq[i]])
    return X_ret, y_ret


X_train_new, y_train_new = rand_seq(X_train_new, y_train_new)


import numpy as np

X_train_new = np.asarray(X_train_new)
X_test = np.asarray(X_test)

import pandas as pd
import keras.layers as layers
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, Flatten
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
import gensim
from gensim.models import FastText
import io


def google_word_vector_model():
    print("Loading GoogleNews-vectors-negative300 ...")
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print("Word2Vec loading complete")
    return model


def train_word2vec_model(sentences, embed_dim):
    print("Training a word2vec model ...")
    model = Word2Vec(sentences, size=embed_dim, workers=2, window=1, min_count=1)
    print("Training complete")
    return model


def facebook_fasttext_model():
    print("Loading Facebook pretrained fasttext model ...")
    fb_model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
    print("Fasttext loading complete")
    return fb_model


def train_fasttext_model(sentences, embed_dim):
    print("Training a fasttext model ...")
    model = FastText(sentences, size=embed_dim, workers=2, window=1, min_count=1)
    print("Training complete")
    return model


def word2vec_embed_layer(word_to_index, sentences):
    vocab_size = 10003
    embed_dim = 300
    #     w2vmodel = train_word2vec_model(sentences, embed_dim)
    w2vmodel = google_word_vector_model()
    #     w2vmodel = facebook_fasttext_model()
    #     w2vmodel = train_fasttext_model(sentences, embed_dim)

    embedding_matrix = np.zeros((vocab_size, embed_dim))
    word_not_exist_count = 0
    for word, i in word_to_index.items():
        try:
            embedding_vector = w2vmodel[word]
        except:
            #             print("Warning: %s not exists in w2vmodel" % word)
            word_not_exist_count += 1
            pass
        try:
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except:
            pass
    print("There're totally %d words not in word2vec." % word_not_exist_count)
    embed_layer = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], trainable=False)
    return embed_layer


from keras.layers import LSTM, Dropout


def classification_model(word_to_index, input_shape, sentences):
    embed_layer = word2vec_embed_layer(word_to_index, sentences)

    input_seq = Input(shape=input_shape)
    embed_seq = embed_layer(input_seq)
    #     x = Dense(256,activation ="relu")(embed_seq)
    x = LSTM(100)(embed_seq)
    x1 = Dropout(0.5)(x)
    x2 = Dense(256, activation='relu')(x1)
    preds = Dense(1, activation="sigmoid")(x2)
    model = Model(input_seq, preds)

    return model


def padding_input(X_train, X_test, maxlen):
#     X_train_pad = pad_sequences(X_train,maxlen=maxlen,padding="post")
#     X_test_pad = pad_sequences(X_test,maxlen=maxlen,padding="post")
    X_train_pad = pad_sequences(X_train,maxlen=maxlen,dtype='float')
    X_test_pad = pad_sequences(X_test,maxlen=maxlen,dtype='float')
    return X_train_pad,X_test_pad

maxlen = max_length + 1 # add <START>
X_train_pad, X_test_pad = padding_input(X_train_new, X_test, maxlen)


model = classification_model(word_index, (X_train.shape[1],), sentences)
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(X_train,y_train2,epochs=20,batch_size=128)

predictions = model.predict(X_test)
# print(predictions)
predictions = [0 if i<0.5 else 1 for i in predictions]
print("Accuracy: ",accuracy_score(y_test2,predictions))
print("Classification Report: ",classification_report(y_test2,predictions))