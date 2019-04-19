"""
Inspired by https://github.com/google/eng-edu/blob/master/ml/guides/text_classification/load_data.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import re
import json

import numpy as np

tweet_jsons = open('tweet.json', 'r')
tweet_ids = open("tweet_id", "r")


def load_tweets():
    seed = 123

    # training data - 80 percent

    train_texts = []
    train_labels = []

    # test data - 20 percent

    test_texts = []
    test_labels = []
    i = 0

    for tweet_json in tweet_jsons.readlines():
        # print("here in json")
        json_version = json.loads(tweet_json)
        # print("json version of id: " + str(json_version["id"]))
        for tweet_id_line in tweet_ids.readlines():
            # print("here in tweetID")
            tweet_id = tweet_id_line.split(",")[0]
            tweet_cat = tweet_id_line.split(",")[1].replace('"', '')

            if str(tweet_id) == str(json_version["id"]):
                # print("here in string equality")
                # print("currentID: " + tweet_id)
                # print(tweet_cat)
                # print(i)
                i = i + 1
                if i <= 3485:
                    train_texts.append(prune_text(json_version["text"]))
                    # print(tweet_cat)
                    # print(str(tweet_cat).strip() == 'n'.strip())
                    # print("\n\n")

                    if str(tweet_cat).strip() == 'n'.strip():
                        train_labels.append(0)
                    else:
                        train_labels.append(1)
                else:
                    test_texts.append(prune_text(json_version["text"]))
                    if str(tweet_cat).strip() == 'n'.strip():
                        test_labels.append(0)
                    else:
                        test_labels.append(1)

                break

        tweet_ids.seek(0)

    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    tweet_jsons.close()

    # print(train_labels)
    # print(test_labels)
    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))


def prune_text(text):
    r = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    new_text = r.sub('HTTPLINK', text)
    text2 = re.sub(r'(^|[^@\w])@(\w{1,15})\b', ' USERNAME ', new_text)
    text3 = text2.replace(",", " ")
    text31 = text3.replace("\n", " ")
    text4 = text31.replace("'", "")
    text5 = text4.replace('"', '')
    text6 = text5.encode("ascii", errors="ignore").decode()
    return text6


# test2 = "Paula, Kiersten and LJ's song about bullying. Great job! (Uploading more videos now.) http://fb.me/ASIm1gw1"
# test3 = "\u3010\u81ea\u52d5post\u3011BULLY\uff08\u3044\u3058\u3081\uff09\u3068SUICIDE\uff08\u81ea\u6bba\uff09\u3067" \
#         "\u3001BULLYCIDE\uff08\u3044\u3058\u3081"
# test4 = "AUISHUAHS eu e o @wallace_mancha tiramos o dia pra praticar bullying com a @helove_"
#
# print(prune_text(test4))

