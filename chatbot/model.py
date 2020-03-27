from __future__ import print_function

import json
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import model_from_json
from . import dataset

class Model:
    model = None
    maxlen = 20
    max_features = 20000
    labels = []
    path = ""

    def __init__(self, path=""):
        self.path = path

    def predict(self, str):
        x_eval = [] 
        x_eval.append(dataset.encode_string(str, num_words=self.max_features))
        x_eval = sequence.pad_sequences(x_eval, maxlen=self.maxlen)

        prediction = self.model.predict(x_eval)[0].argmax()
        return self.labels[prediction]

    def save(self):
        model_json = self.model.to_json()
        saveData = {
            'maxlen': self.maxlen,
            'features': self.max_features,
            'labels': self.labels
        }

        with open(self.path + 'model.json', "w") as json_file:
            json_file.write(json.dumps(saveData))

        with open(self.path + 'model_desc.json', "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(self.path + 'model.h5')

    def load(self):
        json_desc_file = open(self.path + 'model_desc.json', 'r')
        json_file = open(self.path + 'model.json', 'r')
        loadedData = json.loads(json_file.read())
        desc = json_desc_file.read()
        json_file.close()
        json_desc_file.close()

        self.model = model_from_json(desc)
        self.maxlen = loadedData['maxlen']
        self.features = loadedData['features']
        self.labels = loadedData['labels']

        self.model.load_weights(self.path + 'model.h5')

    def compile(self):
        self.model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    def train(self):
        # Embedding
        
        embedding_size = 128

        # Convolution
        kernel_size = 5
        filters = 64
        pool_size = 4

        # LSTM
        lstm_output_size = 70

        # Training
        batch_size = 30
        epochs = 200

        (x_train, y_train, self.labels) = dataset.load_data(self.path + 'data.json', num_words=self.max_features)
        print(len(x_train), 'train sequences')

        print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        print('x_train shape:', x_train.shape)

        print('Build model...')
        self.model = Sequential()
        self.model.add(Embedding(self.max_features, embedding_size, input_length=self.maxlen))
        self.model.add(Dropout(0.25))
        self.model.add(Conv1D(filters,
                        kernel_size,
                        padding='valid',
                        activation='relu',
                        strides=1))
        self.model.add(MaxPooling1D(pool_size=pool_size))
        self.model.add(LSTM(lstm_output_size))
        self.model.add(Dense(y_train.shape[1]))
        self.model.add(Activation('sigmoid'))

        self.compile()

        print('Train...')
        self.model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs)

