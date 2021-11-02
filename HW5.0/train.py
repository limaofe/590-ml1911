import numpy as np
import json
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
from keras.layers import Embedding, SimpleRNN
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import metrics
from keras.regularizers import l2
from keras.regularizers import l1

# load the saved data
data=np.load('data.npz')
data.files
x_train=data['arr_0']
y_train=data['arr_1']
x_val=data['arr_2']
y_val=data['arr_3']

# load the saved word index for reference
f=open('word index.json', 'r', encoding = 'utf-8')
word_index=json.load(f)
f.close()

# set hyper-parameters
max_features=10000
max_len=100

# build 1d convnet
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
# include some form of regularization
model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001)))

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=[metrics.AUC()])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save('1d CNN model.h5')
history.history

# plotting results
auc = history.history['auc_22']
val_auc = history.history['val_auc_22']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(auc) + 1)
plt.plot(epochs, auc, 'bo', label='Training auc')
plt.plot(epochs, val_auc, 'b', label='Validation auc')
plt.title('Training and validation auc')

plt.legend()
plt.show()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')

plt.legend()
plt.show()


# Simple RNN
# stack several recurrent layers one after the other in order to increase the representational power of a network.
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
# include l2 regularization
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001)))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=[metrics.AUC()])

model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save('simple rnn model.h5')

# plotting results
auc = history.history['auc_28']
val_auc = history.history['val_auc_28']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(auc) + 1)
plt.plot(epochs, auc, 'bo', label='Training auc')
plt.plot(epochs, val_auc, 'b', label='Validation auc')
plt.title('Training and validation auc')

plt.legend()
plt.show()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')

plt.legend()
plt.show()
