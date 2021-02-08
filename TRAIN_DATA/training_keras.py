#!/usr/bin/env python3
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
import numpy as np 
from sklearn.model_selection import train_test_split
# from keras.utils import plot_model
from keras.utils import to_categorical

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

max_features = 20
seq_length = 8
batch_size = 32


archs = {'aarch64-rp3': 0,
'alphaev56': 1,
'alphaev67': 2,
'armv8-rp3': 3,
'avr': 4,
'mips': 5,
'mips64el': 6,
'mipsel': 7,
'nios2': 8,
'powerpc': 9  ,
'powerpc64': 10,
'powerpc64le': 11,
'riscv64': 12,
's390': 13,
's390x-64': 14,
 'sh': 15,
'sparc': 16,
'sparc64': 17,
'x86_64-ubuntu18.04-linux-gnu': 18,
'xtensa': 19}

def pad(seq):
    return seq + [0]*(seq_length - len(seq))

def process(f):
    data =   []
    labels = []
    longdata = f.read().strip().split('\n')
    _bool = False
    for i in longdata:
        temp = i.strip().split(' ')
        label = temp[:1][0]
        t = temp[1:]
        
        t = list(map(lambda x: int(x, 16), t))

        for start in range(0, len(t), seq_length):
            # print(start, start + seq_length)
            temp = t[start: start + seq_length]
            if temp == []:
                break 
            if len(temp) < seq_length:
                temp = pad(temp)

            assert(len(temp) == seq_length)
            data.append(temp)
            labels.append(archs[label])

    # print(data[:10])
    # print(labels[:10])
    # print(len(data), len(labels))
    return labels, data 

def read_train_data():
    train_labels = []
    train_data  = []
    _bool = False
    for file in archs:
        # print(file)
        with open('{}.train'.format(file), 'r') as f:
            labels, data = process(f)
            train_data += data
            train_labels += labels
            # train_data.append(data)
            # train_labels.append(labels)

    print(len(data), len(labels), '============')
    # print(train_data[:1])
    # print(train_labels[:1])
    return train_data, train_labels

train_data, train_label = read_train_data()

assert( len(train_data) == len(train_label) )
print(len(train_label), len(train_data))


train_data = np.array(train_data)
train_label = np.array(train_label)

print(train_label.shape, train_data.shape)
print(train_data[:3])
print(train_label[:3])
# train_label = to_categorical(train_label, num_classes=None)

x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2) 

print('Train size: {} {}'.format(len(x_train), len(y_train)))

print('Test size: {} {}'.format(len(x_test), len(y_test)))

model = Sequential()
