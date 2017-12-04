import wave
import numpy as np
import math
import os
import pickle
from sklearn.ensemble import RandomForestRegressor as RFR
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import TensorBoard

# train LSTM NN
def train_lstm(x, y, xt, yt):
    batch_size = 320
    nb_classes = 10
    nb_epoch = 15
    ts = x[0].shape[0]
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(ts, x[0].shape[1])))
    model.add(Activation('tanh'))
    # model.add(LSTM(512, return_sequences=True))
    # model.add(Activation('tanh'))
    model.add(LSTM(256, return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dense(512))
    model.add(Activation('tanh'))
    model.add(Dense(y.shape[1]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    model.fit(x, y,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=[xt, yt])
              #callbacks=[TensorBoard(log_dir='./training_logs',histogram_freq=1,write_grads=True,write_images=True)])
    loss, acc = model.evaluate(xt, yt, batch_size=batch_size, verbose=1)
    print(loss, acc)
    print(model.summary())
    return model


def make_sample_lstm(x, n, y=None, use_y=False):
    if use_y:
        xt = np.zeros((x.shape[0], n, x.shape[1] + 1), dtype=float)
        t = np.zeros((x.shape[0], x.shape[1] + 1), dtype=float)
        for i in range(x.shape[0]):
            if i == 0:
                t[i, :-1] = x[i, :]
                t[i, -1] = 0
            else:
                t[i, :-1] = x[i, :]
                t[i, -1] = y[i - 1]

        for i in range(x.shape[0]):
            if i < n:
                i0 = n - i
                xt[i, :i0, :] = np.zeros((i0, x.shape[1] + 1), dtype=float)
                if i > 0:
                    xt[i, i0:, :] = t[:i, :]
            else:
                xt[i, :, :] = t[i - n:i, :]
        return xt
    else:
        xt = np.zeros((x.shape[0], n, x.shape[1]), dtype=float)
        for i in range(x.shape[0]):
            if i < n:
                i0 = n - i
                xt[i, :i0, :] = np.zeros((i0, x.shape[1]), dtype=float)
                if i > 0:
                    xt[i, i0:, :] = x[:i, :]
            else:
                xt[i, :, :] = x[i - n:i, :]
        return xt


class modelLSTM:
    def __init__(self, model, length, use_y):
        self.model = model
        self.n = length
        self.use_y = use_y

    def predict(self, x):
        if self.use_y:
            result = np.zeros((x.shape[0], 1), dtype=float)
            for i in range(x.shape[0]):
                t = np.zeros((self.n, x.shape[1] + 1), dtype=float)
        else:
            xt = make_sample_lstm(x, self.n)
            return self.model.predict(xt)

    def save(self, name_json, name_weights):
        json_string = self.model.to_json()
        open(name_json, 'w').write(json_string)
        self.model.save_weights(name_weights)


def train_lstm_avec(x, y, xt, yt):
    length = 25
    use_y = False

    x_series_train = make_sample_lstm(x, length, y, use_y)
    print(x_series_train.shape)
    x_series_test = make_sample_lstm(xt, length, yt, use_y)
    print(x_series_test.shape)

    print(y[:100, 0])
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(x_series_train.shape[1], x_series_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Activation('tanh'))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Activation('tanh'))
    # model.add(Dense(128))
    # model.add(Activation('tanh'))
    model.add(Dense(1))
    # model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='mean_absolute_error', optimizer='rmsprop')
    model.fit(x_series_train, y, batch_size=512, nb_epoch=50,
              verbose=2, validation_data=(x_series_test, yt))

    return modelLSTM(model, length, use_y)


# train multilayer perceptron
def train_mpc(x, y, tx, ty):
    batch_size = 256
    nb_classes = 10
    nb_epoch = 20

    model = Sequential()
    model.add(Dense(512, input_shape=(x.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(y.shape[1]))
    # model.add(Activation('softmax'))

    # model.summary()

    model.compile(loss='mean_absolute_error',
                  optimizer=RMSprop())

    history = model.fit(x, y,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=0, validation_data=(tx, ty), show_accuracy=True)

    return model


# train Random Forest classifier
def train_rfr(x, y, tx, ty):
    rfr = RFR(n_estimators=50)
    model = rfr.fit(x, y)
    return model


def train_rfc(x, y, options):
    classifier = RFR(n_estimators=100)
    return classifier.fit(x, y)


def train(x, y, options):
    classes = np.unique(y)
    print(classes)

    classifiers = []
    for c in classes:
        out = np.array([int(i) for i in y == c])
        print(c, len(out[out == 1]), len(out[out == 0]), len(out[out == 1]) + len(out[out == 0]))

        classifier = RFR(n_estimators=10)
        classifier.fit(x, out)
        classifiers.append(classifier)
    return classifiers











