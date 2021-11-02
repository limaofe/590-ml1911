import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

m=2000 # set 1000 samples per novel

# first novel
# read the text file
sourcefile='The Game of Go by Arthur Smith.txt'
data = open(sourcefile,'r',encoding='utf-8')
ListOfLine = data.read().splitlines()

# delete the irrelavent lines
ListOfLine=ListOfLine[29:]
# delete the blank lines
while '' in ListOfLine:
    ListOfLine.remove('')

# determine how many lines could consist of a file
l=len(ListOfLine)//m+1

# break the novels into chunks of text 
trainsource='train\The Game of Go by Arthur Smith\The Game of Go by Arthur Smith'
testsource='test\The Game of Go by Arthur Smith\The Game of Go by Arthur Smith'
for i in range(m):
    print("Initialize no."+str(i+1)+' file')
    if i<1600:
        trainFileName = os.path.splitext(trainsource)[0]+"_part"+str(i)+".txt" 
        trainFileData = open(trainFileName,"w",encoding='utf-8')
        for line in ListOfLine[i*l:(i+1)*l]:
            trainFileData.write(line+'\n')
        trainFileData.close()
    else:
        testFileName = os.path.splitext(testsource)[0]+"_part"+str(i)+".txt" 
        testFileData = open(testFileName,"w",encoding='utf-8')
        if(i==m-1):
            for line in ListOfLine[i*l:]:
                testFileData.write(line+'\n')
        else:
            for line in ListOfLine[i*l:(i+1)*l]:
                testFileData.write(line+'\n')
        testFileData.close()
    print("End")
    
# second novel
# read the text file
sourcefile='After the Manner of Men.txt'
data = open(sourcefile,'r',encoding='utf-8')
ListOfLine = data.read().splitlines()

# delete the irrelavent lines
ListOfLine=ListOfLine[31:]
# delete the blank lines
while '' in ListOfLine:
    ListOfLine.remove('')

# determine how many lines could consist of a file
l=len(ListOfLine)//m+1

# break the novels into chunks of text 
trainsource='train\After the Manner of Men\After the Manner of Men'
testsource='test\After the Manner of Men\After the Manner of Men'
for i in range(m):
    print("Initialize no."+str(i+1)+' file')
    if i<1600:
        trainFileName = os.path.splitext(trainsource)[0]+"_part"+str(i)+".txt" 
        trainFileData = open(trainFileName,"w",encoding='utf-8')
        for line in ListOfLine[i*l:(i+1)*l]:
            trainFileData.write(line+'\n')
        trainFileData.close()
    else:
        testFileName = os.path.splitext(testsource)[0]+"_part"+str(i)+".txt" 
        testFileData = open(testFileName,"w",encoding='utf-8')
        if(i==m-1):
            for line in ListOfLine[i*l:]:
                testFileData.write(line+'\n')
        else:
            for line in ListOfLine[i*l:(i+1)*l]:
                testFileData.write(line+'\n')
        testFileData.close()
    print("End")

# third novel
# read the text file
sourcefile='Honor of Thieves.txt'
data = open(sourcefile,'r',encoding='utf-8')
ListOfLine = data.read().splitlines()

# delete the irrelavent lines
ListOfLine=ListOfLine[30:]
# delete the blank lines
while '' in ListOfLine:
    ListOfLine.remove('')

# determine how many lines could consist of a file
l=len(ListOfLine)//m+1

# break the novels into chunks of text 
trainsource='train\Honor of Thieves\Honor of Thieves'
testsource='test\Honor of Thieves\Honor of Thieves'
for i in range(m):
    print("Initialize no."+str(i+1)+' file')
    if i<1600:
        trainFileName = os.path.splitext(trainsource)[0]+"_part"+str(i)+".txt" 
        trainFileData = open(trainFileName,"w",encoding='utf-8')
        for line in ListOfLine[i*l:(i+1)*l]:
            trainFileData.write(line+'\n')
        trainFileData.close()
    else:
        testFileName = os.path.splitext(testsource)[0]+"_part"+str(i)+".txt" 
        testFileData = open(testFileName,"w",encoding='utf-8')
        if(i==m-1):
            for line in ListOfLine[i*l:]:
                testFileData.write(line+'\n')
        else:
            for line in ListOfLine[i*l:(i+1)*l]:
                testFileData.write(line+'\n')
        testFileData.close()
    print("End")
    


labels = []
texts = []
for label_type in ['After the Manner of Men', 'Honor of Thieves', 'The Game of Go by Arthur Smith']:
    dir_name = os.path.join('train', label_type)
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
training_samples = 4000
validation_samples = 800
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

# split into train and val datasets
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:]
y_val = labels[training_samples:]

# save the processed data
np.savez('data', x_train, y_train, x_val, y_val)

jsObj = json.dumps(word_index)
fileObject = open('word index.json', 'w')  
fileObject.write(jsObj)  
fileObject.close()  
