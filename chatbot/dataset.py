from __future__ import print_function
from keras.preprocessing.text import hashing_trick
import numpy as np 
import json

def encode_string(text, num_words=2000):
    return hashing_trick(text, num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', hash_function='md5')

def load_data(path="data.json", num_words=2000, maxlen=None, seed=113):
    with open(path, encoding="utf8") as data_file:
        data = json.load(data_file)
        labels = dict()
        ret_labels = []
        for p in data:
            labels[p['tag']] = len(labels)
            ret_labels.append(p['tag'])
        
        y_train = []
        x_train = []
        y_train_dim = len(labels)
        y_placeholder = [0 for x in range(y_train_dim)]

        for d in data:
            for p in d['patterns']:
                x_train.append(encode_string(p, num_words=num_words))
                y = labels[d['tag']]
                y_data = y_placeholder.copy()
                y_data[y] = 1
                y_train.append(y_data)

        y_train = np.reshape(y_train, (-1, y_train_dim))

        return (x_train, y_train, ret_labels)
