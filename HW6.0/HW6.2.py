from keras.datasets import mnist, fashion_mnist
from keras import layers, Input
import keras
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
N_channels=1; PIX=28
# Normalize into [0,1]
x_train = x_train.astype('float32') / 255. 
x_test = x_test.astype('float32') / 255.
# Downsize the dateset
x_train=x_train[:30000]
x_test=x_test[:30000]

input_img = Input(shape=(PIX, PIX, N_channels))

# #ENCODER
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
# # AT THIS POINT THE REPRESENTATION IS (4, 4, 8) I.E. 128-DIMENSIONAL
 
# Decoder
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy');
autoencoder.summary()

# Train the model
history = autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=1000,
                shuffle=True,
                validation_data=(x_test, x_test),
                )

#HISTORY PLOT
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.legend()

#MAKE PREDICTIONS FOR TEST DATA
decoded_imgs = autoencoder.predict(x_test)

#VISUALIZE THE RESULTS
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(PIX, PIX,N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(PIX, PIX,N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# Prepare the MNIST-FASHION dataset as anomalies 
(_, _), (x_abnormal, _) = fashion_mnist.load_data()
x_abnormal = x_abnormal / 255.


# Set the threshold for anomalies
threshold = 2 * autoencoder.evaluate(x_train, x_train)

# Loop through the dataset to find the number of anomalies that is beyond the threshold
predict_result = autoencoder.predict(x_abnormal)
counter = 0  
for i in range(x_abnormal.shape[0]):
    error=np.mean((x_abnormal[i] - predict_result[i])**2)  
    if error>threshold:
        counter += 1
print('The number of anomalies in MNIST-FASHION is: ', counter, '. The fraction of times anomalies are detected is ', counter/x_abnormal.shape[0], sep='')

# Repeat the step to MNIST test set
predict_result = autoencoder.predict(x_test)
counter = 0  
for i in range(x_abnormal.shape[0]):
    error=np.mean((x_test[i] - predict_result[i])**2)  
    if error>threshold:
        counter += 1
print('The number of anomalies in MNIST is: ', counter, '. The fraction of times anomalies are detected is ', counter/x_abnormal.shape[0], sep='')

# We can find that the number of anomalies in MNIST is much lower than in MNIST-FASHION