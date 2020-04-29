#!/usr/bin/env python3
from pre_processing import *


X_train, Y_train = generate_data_flatten()

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

print('Test accuracy:', score[1])


