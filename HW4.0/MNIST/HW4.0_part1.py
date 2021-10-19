from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import to_categorical
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import rescale, resize, downscale_local_mean
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

# load three datasets into a list
# use index to access specific dataset 0 for mnist, 1 for fashion_mnist, 2 for cifar10

datasets = [mnist.load_data(),fashion_mnist.load_data(),cifar10.load_data()]

# hyper-param
dataset='fashion_mnist' # initialize the variable to choose dataset
choose_model='CNN' # choose model from CNN and DFF
batch_size=64
epochs=10
data_augmentation='T' # whether to implement data augmentation or not
# optimizer='rmsprop' # we can test different optimizer
# manual hyper-parameter tuning with different optimizer
optimizer = SGD(0.01) # We’ll use the SGD optimizer to train the network with a learning rate of 0.01
loss='categorical_crossentropy'
metrics=['accuracy']


# split the dataset
if dataset=='mnist':
    (x_train, y_train), (x_test, y_test)=datasets[0]
elif dataset=='fashion_mnist':
    (x_train, y_train), (x_test, y_test)=datasets[1]
elif dataset=='cifar10':
    (x_train, y_train), (x_test, y_test)=datasets[2]
else:
    print('You may put a wrong value of dataset variable.')
    
# show the surface plot
def surface_plot(image):
    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    fig = plt.figure()
    ax = fig.gca(projection='3d') #viridis
    ax.plot_surface(xx, yy, image[:,:] ,rstride=1, cstride=1, cmap=plt.cm.gray,linewidth=0)
    plt.show()
surface_plot(x_train[0])

# Add function to visualize a random (or specified) image in the dataset
def showimage(images, num):
    ind=np.random.randint(0, len(images)-1, num)
    for i in ind:
        image=images[i]
        image = resize(image, (10, 10), anti_aliasing=True)
        plt.imshow(image, cmap=plt.cm.gray); plt.show()
# visualize 5 random image in the dataset
showimage(x_train, 5)

# save the basic information 
train=len(x_train)
test=len(x_test)
height=x_train.shape[1]
width=x_train.shape[2]
try:
    channel=x_train.shape[3]
except:
    channel=1

# implement data augmentatio
if data_augmentation=='T' and choose_model=='CNN':
# set up image augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
        #zoom_range=0.3
        )
    x_train=x_train.reshape((train, height, width, channel))
    datagen.fit(x_train)

######################### Dense Feed forward ANN #########################
#INITIALIZE MODEL	
	# Sequential model --> plain stack of layers 	
	# each layer has exactly one input tensor and one output tensor.
if choose_model=='DFF':
    model = models.Sequential()
    
    #ADD LAYERS
    model.add(layers.Dense(1024, activation='relu', input_shape=(height * width * channel, )))
    #ADD LAYERS
    model.add(layers.Dense(512, activation='relu'))
    #SOFTMAX  --> 10 probability scores (summing to 1
    model.add(layers.Dense(10,  activation='softmax'))
    
    #COMPILATION (i.e. choose optimizer, loss, and metrics to monitor)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    
    #PREPROCESS THE DATA
    x_train1 = x_train.reshape((train, height * width * channel,)) 
    #RESCALE INTS [0 to 255] MATRIX INTO RANGE FLOATS RANGE [0 TO 1] 
    x_train1 = x_train1.astype('float32') / 255
    
    #REPEAT FOR TEST DATA
    x_test1 = x_test.reshape((test, height * width * channel,))
    x_test1 = x_test1.astype('float32') / 255
    
    
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    
    # Do 80-20 split of the “training” data into (train/validation)
    val_size=int(len(x_train1)*0.2)
    x_val=x_train1[:val_size]
    x_train1=x_train1[val_size:]
    y_val=y_train1[:val_size]
    y_train1=y_train1[val_size:]
    
    #TRAIN 
    history=model.fit(x_train1, y_train1, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))


########################### A convolutional NN ###########################
if choose_model=='CNN':
    # instantiating a covnet
    model = models.Sequential()
    # the format of image is (28, 28, 1) so set the input_shape with the same value
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channel)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    # preprocess data
    
    # transform the train and test sets as the input format
    x_train1 = x_train.reshape((train, height, width, channel))
    x_train1 = x_train1.astype('float32') / 255
    x_test1 = x_test.reshape((test, height, width, channel))
    x_test1 = x_test1.astype('float32') / 255
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    
    # Do 80-20 split of the “training” data into (train/validation)
    val_size=int(len(x_train1)*0.2)
    x_val=x_train1[:val_size]
    x_train1=x_train1[val_size:]
    y_val=y_train1[:val_size]
    y_train1=y_train1[val_size:]
    
    # compile the model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    # train the model
    history=model.fit(x_train1, y_train1, epochs=epochs, batch_size=64, validation_data=(x_val, y_val))

# Include a training/validation history plot
# accuracy plot
plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# loss plot
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

if data_augmentation=='T' and choose_model=='CNN':
    # fit model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channel)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    # transform the train and test sets as the input format
    x_train1 = x_train.reshape((train, height, width, channel))
    x_train1 = x_train1.astype('float32') / 255
    x_test1 = x_test.reshape((test, height, width, channel))
    x_test1 = x_test1.astype('float32') / 255
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    
    # Do 80-20 split of the “training” data into (train/validation)
    val_size=int(len(x_train1)*0.2)
    x_val=x_train1[:val_size]
    x_train1=x_train1[val_size:]
    y_val=y_train1[:val_size]
    y_train1=y_train1[val_size:]

    history2=model.fit_generator(datagen.flow(x_train1, y_train1, batch_size=128),
                    steps_per_epoch = len(x_train1) / 128, epochs=epochs, validation_data=(x_val, y_val))



# evaluate the model
train_loss, train_acc = model.evaluate(x_train1, y_train1, batch_size=batch_size)
test_loss, test_acc = model.evaluate(x_test1, y_test1, batch_size=batch_size)
val_loss, val_acc = model.evaluate(x_val, y_val, batch_size=batch_size)
print('train_acc:', train_acc)
print('test_acc:', test_acc)
print('val_acc:', val_acc)


# Include a method to save model and hyper parameters
model.save('my_model.h5')
# Include a method to read a model from a file
model_from_file=load_model('my_model.h5')




