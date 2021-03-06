from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, LSTM, TimeDistributed, Reshape
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train)
# print(y_train)
# print(x_train.shape)  # (60000, 28, 28)
# print(y_train.shape)  # (60000,)

x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')/255

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(x_train.shape)  # (60000, 28, 28, 1)
# print(y_train.shape)  # (60000, 10)


model = Sequential() 
model.add(Conv2D( 7 , (2,2) , padding = 'valid',      # padding = 'valid' : 패딩 안 한다.
                 input_shape = (28,28,1), strides = 2))
model.add(Conv2D(100,(2,2)))
model.add(MaxPooling2D(2,2))
#model.summary()
model.add(Reshape((36,100)))
model.add(LSTM(32))
model.add(Dense(10, activation = 'softmax'))

####################TimeDistributed활용####################################################
# model = Sequential() 
# model.add(Conv2D( 7 , (2,2) , padding = 'valid',      # padding = 'valid' : 패딩 안 한다.
#                  input_shape = (28,28,1), strides = 2))
# model.add(Conv2D(100,(2,2)))
# model.add(MaxPooling2D(2,2))
# model.add(TimeDistributed(Flatten()))
# model.add(LSTM(256, return_sequences=True))
# model.add(TimeDistributed(Dense(8), input_shape = (28,28)))
# model.add(Flatten())
# model.add(Dense(10, activation = 'softmax'))
############################################################################################

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=20)

model.fit(x_train, y_train, validation_split = 0.2,
          epochs=100, batch_size=8, verbose = 1,
          callbacks=[early_stopping])

acc = model.evaluate(x_test, y_test)

print(acc)
