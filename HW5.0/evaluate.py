import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
from keras.layers import Embedding, SimpleRNN
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import metrics
from keras.regularizers import l2
from keras.regularizers import l1
from keras.models import load_model
labels = []
texts = []
for label_type in ['After the Manner of Men', 'Honor of Thieves', 'The Game of Go by Arthur Smith']:
    dir_name = os.path.join('test', label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding='utf-8')
            texts.append(f.read())
            f.close()
        if label_type == 'The Game of Go by Arthur Smith':
            labels.append(0)
        elif label_type == 'After the Manner of Men':
            labels.append(1)
        else:
            labels.append(2)
            
# Tokenizing the text of the raw data
maxlen = 100
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

model=load_model('simple rnn model.h5')
model.evaluate(data, labels)
