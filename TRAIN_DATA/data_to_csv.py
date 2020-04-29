#!/usr/bin/env python3

from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
import numpy as np 

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

def find_max_seq(header):
    t = 0 
    for i in header:
        for j in header[i]:
            tt = len(j)
            if t < tt:
                t = tt
                mark = j 
    return t, j

def my_padding_process(arch):
    result = []
    with open("{}.train".format(arch), 'r') as f:
        data = f.read().strip()
        data = data.replace("{} ".format(arch), '')
        data = data.split('\n')
        for d in data:
            for i in range(0, len(d), SIZE):
                pad_i = d[i: i + SIZE].split(' ')
                if len(pad_i) < SIZE:
                    # Only pad sequence that has data from 2-16
                    if  len(pad_i) < SIZE/8:
                        # print("break")
                        # print(pad_i, len(pad_i))
                        break
                    else:
                        pad_i = pad_i + ['00']*(SIZE - len(pad_i))
                assert(len(pad_i) == SIZE)
                # print(pad_i)
                space_i = ' '.join(pad_i)
                # print(space_i)
                # space_i = one_hot(space_i, n=2**16, split=' ')
                result.append(space_i)
        return result 

def keras_padding_process(arch):
    result = []
    with open("{}.train".format(arch), 'r') as f:
        data = f.read().strip()
        data = data.replace("{} ".format(arch), '')
        data = data.split('\n')
        return data 

def generate_data():
    X_train, Y_train = [], [] 
    for arch in archs:
        temp = keras_padding_process(arch)
        X_train.append( temp )
        Y_train.append( ['{}'.format(arch)]*len(temp) )
    return X_train, Y_train

X_train, Y_train = generate_data()

NUM_WORDS = 10000
# Tokenize the input 
x = np.array(X_train) #, dtype=np.int32)
y = np.array(Y_train) #, dtype=np.int32)

tokenizerx = Tokenizer(num_words=NUM_WORDS)
tokenizerx.fit_on_texts(X_train)
train_data = tokenizerx.texts_to_sequences(X_train)

tokenizery = Tokenizer(num_words=NUM_WORDS)
tokenizery.fit_on_texts(Y_train)
train_label = tokenizery.texts_to_sequences(Y_train)

x_train = np.array(train_data)
y_train = np.array(train_label)

# work around
new_y = [] 
for i in range(len(x_train)):
    t = len(x_train[i])
    new_y.append( [y_train[i][0]]*t )


new_y = np.array(new_y)
vocab_size = len(tokenizerx.word_index) + 1

# Save some memory
# del y_train, Y_train

from keras.models import Sequential
from keras import layers
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Activation
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
x_binary = to_categorical(x_train)
# y_binary = to_categorical(new_y)


x_train, x_test, y_train, y_test = train_test_split(x_train, new_y , test_size=0.2)

batch_size = 64

model = Sequential()
model.add(Dense(512, input_dim=1))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('softmax'))
model.summary()
 
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=30,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
 
print('Test accuracy:', score[1])
 