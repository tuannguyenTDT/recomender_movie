
from sklearn.metrics import mean_absolute_error
import pickle

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dropout
from keras.layers.merge import Dot, multiply, concatenate
from keras.models import load_model
from collections import defaultdict

def get_mapping(series):
    occurances = defaultdict(int)
    for element in series:
        occurances[element] += 1
    mapping = {}
    i = 0
    for element in occurances:
        i += 1
        mapping[element] = i

    return mapping




def get_data():
    data = pd.read_csv("../ml-latest-small/ratings.csv")

    mapping_work = get_mapping(data["movieId"])

    data["movieId"] = data["movieId"].map(mapping_work)

    mapping_users = get_mapping(data["movieId"])

    data["movieId"] = data["movieId"].map(mapping_users)

    percentil_80 = np.percentile(data["timestamp"], 80)

    print(percentil_80)

    print(np.mean(data["timestamp"]<percentil_80))

    print(np.mean(data["timestamp"]>percentil_80))

    cols = ["userId", "movieId", "rating"]

    train = data[data.timestamp<percentil_80][cols]

    print(train.shape)

    test = data[data.timestamp>=percentil_80][cols]

    print(test.shape)

    max_user = max(data["userId"].tolist() )
    max_work = max(data["movieId"].tolist() )


    return train, test, max_user, max_work, mapping_work




def get_model_1(max_work, max_user):
    dim_embedddings = 30
    bias = 3
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    o = multiply([w, u])
    o = Dropout(0.5)(o)
    o = Flatten()(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    rec_model.summary()
    rec_model.compile(loss='mae', optimizer='adam', metrics=["mae"])

    return rec_model


def get_model_2(max_work, max_user):
    dim_embedddings = 30
    bias = 1
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)
    w_bis = Embedding(max_work + 1, bias, name="workbias")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    u_bis = Embedding(max_user + 1, bias, name="userbias")(u_inputs)
    o = multiply([w, u])
    o = concatenate([o, u_bis, w_bis])
    o = Dropout(0.5)(o)
    o = Flatten()(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    rec_model.summary()
    rec_model.compile(loss='mae', optimizer='adam', metrics=["mae"])

    return rec_model

def get_model_3(max_work, max_user):
    dim_embedddings = 30
    bias = 1
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)
    w_bis = Embedding(max_work + 1, bias, name="workbias")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    u_bis = Embedding(max_user + 1, bias, name="userbias")(u_inputs)
    o = multiply([w, u])
    o = Dropout(0.5)(o)
    o = concatenate([o, u_bis, w_bis])
    o = Flatten()(o)
    o = Dense(10, activation="relu")(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    rec_model.summary()
    rec_model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

    return rec_model

def get_array(series):
    return np.array([[element] for element in series])

train, test, max_user, max_work, mapping_work = get_data()


model = get_model_3(max_work, max_user)

history = model.fit([get_array(train["movieId"]), get_array(train["userId"])], get_array(train["rating"]), nb_epoch=2,
                    validation_split=0.2)
# model.save('model_1.h5')
#
# model_trained = load_model('model_1.h5')

predictions = model.predict([get_array(test["movieId"]), get_array(test["userId"])])


predictions = model.predict([get_array(test["movieId"]), get_array(test["userId"])])
print(predictions)

print("Gia tri thuc")

print(get_array(test['rating']))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

