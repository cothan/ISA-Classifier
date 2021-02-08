#!/usr/bin/env python3

import numpy as np
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Activation
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


SIZE = 16

archlist = ['aarch64-rp3',
            'alphaev56',
            'alphaev67',
            'armv8-rp3',
            'mips',
            'mips64el',
            'mipsel',
            'powerpc',
            'powerpc64le',
            'riscv64',
            's390',
            's390x-64',
            'sh',
            'sparc',
            'sparc64',
            'x86_64-ubuntu18.04-linux-gnu',
            'xtensa']

archs = {}
for i in range(len(archlist)):
    archs[archlist[i]] = i

def my_padding_process(arch):
    result = []
    with open("{}.train".format(arch), 'r') as f:
        data = f.read().strip()
        data = data.replace("{} ".format(arch), '')
        data = data.split('\n')
        for d in data[:10]:
            d = d.split(" ")
            for i in range(0, len(d), SIZE):
                pad_i = d[i: i + SIZE]
                # print(pad_i)
                if len(pad_i) < SIZE:
                    pad_i = pad_i + ['00']*(SIZE - len(pad_i))
                assert(len(pad_i) == SIZE)
                # print(pad_i)
                space_i = ' '.join(pad_i)
                # print(space_i)
                # space_i = one_hot(space_i, n=2**16, split=' ')
                result.append(space_i)
        return result

def generate_data_flatten():
    X_train, Y_train = [], []
    for arch in archs:
        temp = my_padding_process(arch)
        X_train += temp
        Y_train += ['{}'.format(arch)]*len(temp)
    return X_train, Y_train


X_train, Y_train = generate_data_flatten()

token = Tokenizer()
token.fit_on_texts(X_train)
# token.fit_on_texts(Y_train)
x_train = token.texts_to_sequences(X_train)
# y_train = token.texts_to_sequences(Y_train)

print('===========================')
print(x_train[0])
print(X_train[0])
print(token.index_word[x_train[0][0]])
print('===========================')
# print(y_train[0])
print(Y_train[0])
# print(token.index_word[y_train[0][0]])

# After tokenize the training data, the x_train has different length, so need to pad them

data = pad_sequences(x_train, padding='post')
label = pad_sequences(Y_train, padding='post')

# No need to pad y_train

a_data, a_test_data, b_label, b_test_label = train_test_split(data, label, test_size=0.3)

batch_size = 64

model = Sequential()
model.add(Dense(512, input_dim=16))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(6))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(a_data, b_label,
                    batch_size=batch_size,
                    epochs=10,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(a_test_data, b_test_label,
                       batch_size=batch_size, verbose=1)


print('Test accuracy:', score)
