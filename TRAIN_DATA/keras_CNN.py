#!/usr/bin/env python3
from pre_processing import *

# Better
choice = 2

if choice == 1:
    X_train, Y_train = generate_data_flatten()
else:
    X_train, Y_train = generate_fix_sequence()

token = Tokenizer()
token.fit_on_texts(X_train)
token.fit_on_texts(Y_train)
x_train = token.texts_to_sequences(X_train)
y_train = token.texts_to_sequences(Y_train)

print('===========================')
print(x_train[0])
print(X_train[0])
print(token.index_word[x_train[0][0]])
print('===========================')
print(y_train[0])
print(Y_train[0])
print(token.index_word[y_train[0][0]])

# After tokenize the training data, the x_train has different length, so need to pad them

data = pad_sequences(x_train, padding='post')
label = pad_sequences(y_train, padding='post')

print('===========================')
print(x_train[0])
print(X_train[0])
print(token.index_word[x_train[0][0]])
print('===========================')
print(y_train[0])
print(Y_train[0])
print(token.index_word[y_train[0][0]])


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
                    epochs=2,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(a_test_data, b_test_label,
                       batch_size=batch_size, verbose=1)

print('Test accuracy:', score)


# model.predict()


x = ['c10183f9027eee83350000000001b800000000c3b800000000eb0383c00183f8027ef866810d000000009100b8faffffffc3415541545553488b1d0000000044',
'000000000000757d0000ffffffffffffffff222122212221326316d432211f93322115d33263322172210263322102851793022132213221322132212221fce2',
'3d0bc60000280738171df0003661007d01290739174927593728273837302220acd2380721000027930d2817cc822827660204283726021828073817410000c8',
'00810a000891090000814a000c914900003d400000814a00009149000038a0000038c000003c600000386300004cc631824800000138600000800100347c0803',
'5810d132502010005820d12e58202000502010005820d12a58202000502010005820d1265820200050201000a72800ff50201000a7380000a7f4001d18138910',
'05b90000000021ca8910488b45e88b0083e00189c2488b45e889100fb645bc5dc3554889e589f98975e889d0884dec8845e448c745f80000000048c745f00000',
'91090000910900009109000091090000908900009149000090e90000914900009149000090e90000914900009149000090e90000914900004cc6318248000001',
'12100000050040100000023c0000023c2000428c120000100000063c0400438c0400428c000000000000428c00000000000062ac0000023c2400428c08000010',
'814b815c816d817e818f819885098500d0a22eb32ec42ed52ee62ef72e082f192f00c0a980ba80cb80dc80ed80fe800f8118852a2c3b2c4c2c5d2c6e2c7f2c80',
'005840b03898beb02c07f4070707070707908ff0200dd0a7faff8818bf5020b06c5030b0685040b0645050b060a71800005010b074a7f400215810b0745a10b0',]

x_text_process = []
for i in range(len(x)):
    x_text_process.append(text_process(x[i]))


# import IPython
# IPython.embed()


