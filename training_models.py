import wave
import numpy as np
import math
import os
import pickle
from sklearn.ensemble import RandomForestRegressor as RFR
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers import LSTM,Conv1D, Conv2D, GlobalMaxPooling1D, MaxPooling1D, TimeDistributed, BatchNormalization,MaxPooling2D,Embedding,SimpleRNN
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import TensorBoard
from keras import regularizers
import matplotlib.pyplot as plt

# train LSTM NN
def train_lstm(x, y, xt, yt):
    batch_size = 512
    nb_epoch = 25
    timestep = x[0].shape[0]
    inputdim=x[0].shape[1]
    outputdim=y.shape[1]

    model = Sequential()
    # model.add(Dense(units=512,input_shape=(timestep, inputdim),activation='relu'))
    # model.add(SimpleRNN(
    #     # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    #     # Otherwise, model.evaluate() will get error.
    #     #batch_input_shape=(None, timestep, inputdim),  # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    #     output_dim=70,
    #     unroll=True,
    # ))
    model.add(LSTM(512,input_shape=(timestep, inputdim), return_sequences=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=False))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(outputdim))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
    history=model.fit(x, y,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=[xt, yt])
              #callbacks=[TensorBoard(log_dir='./training_logs',histogram_freq=1,write_grads=True,write_images=True)])

    loss, acc = model.evaluate(xt, yt, batch_size=batch_size, verbose=1)
    print(loss, acc)
    print(model.summary())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.show()
    return model

def train_cnn(x_train, y_train, x_test, y_test):
    batch_size = 128
    print('Loading data...')
    print(x_train[0].shape[0])
    model = Sequential()
    model.add(Conv1D(32, kernel_size=4, activation='relu', padding='same',input_shape=(x_train[0].shape[0], x_train[0].shape[1])))
    model.add(MaxPooling1D(4))
    model.add(Conv1D(64, 4, activation='relu', padding='same'))
    model.add(MaxPooling1D(4))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.2))
    #
    # model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.2))
    #
    # model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
    # model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['categorical_accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=100,
                        shuffle=True,
                        validation_data=(x_test, y_test))

    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

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
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    model.fit(x_series_train, y, batch_size=512, nb_epoch=15,
              verbose=2, validation_data=(x_series_test, yt))

    return modelLSTM(model, length, use_y)

# train multilayer perceptron
def train_mpc(x, y, tx, ty):
    batch_size = 256
    nb_epoch = 20
    timestep = x[0].shape[0]
    inputdim = x[0].shape[1]

    model = Sequential()
    model.add(Dense(512, input_shape=(timestep,inputdim)))
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
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.summary()
    model.compile(loss='mean_absolute_error',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x, y,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(tx, ty))

    return model














