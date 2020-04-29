
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
                space_i = ''.join(pad_i)
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