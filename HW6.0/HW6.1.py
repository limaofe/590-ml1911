import matplotlib.pyplot as plt
from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize into [0,1]
x_train = x_train / 255. 
x_test = x_test / 255.
x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

# Set the bottleneck in the network which forces a compressed knowledge representation of the original input
n_bottleneck=48

# Set the input shape of image
input_shape=(28*28, )

# Build the autoencoder
ae=Sequential()            
ae.add(Dense(n_bottleneck, activation='relu', input_shape=input_shape))
ae.add(Dense(784, activation='sigmoid'))

# Compile the autoencoder and fit it
ae.compile(optimizer='adam',
           loss='mean_squared_error')
ae.summary()
history=ae.fit(x=x_train,
       y=x_train,
       epochs=20,
       batch_size=1000,
       validation_split=0.2
       )
history.history
ae.save('Autoencoder.h5')

# Plotting results
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')

plt.legend()
plt.show()

# Prepare the MNIST-FASHION dataset as anomalies 
(_, _), (x_abnormal, _) = fashion_mnist.load_data()
x_abnormal = x_abnormal / 255.
x_abnormal = x_abnormal.reshape(-1,28*28)


# Set the threshold for anomalies
threshold = 4 * ae.evaluate(x_train, x_train)

# Loop through the dataset to find the number of anomalies that is beyond the threshold
predict_result = ae.predict(x_abnormal)
counter = 0  
for i in range(x_abnormal.shape[0]):
    error=np.mean((x_abnormal[i] - predict_result[i])**2)  
    if error>threshold:
        counter += 1
print('The number of anomalies in MNIST-FASHION is: ', counter, '. The fraction of times anomalies are detected is ', counter/x_abnormal.shape[0], sep='')

# Repeat the step to MNIST test set
predict_result = ae.predict(x_test)
counter = 0  
for i in range(x_abnormal.shape[0]):
    error=np.mean((x_test[i] - predict_result[i])**2)  
    if error>threshold:
        counter += 1
print('The number of anomalies in MNIST is: ', counter, '. The fraction of times anomalies are detected is ', counter/x_abnormal.shape[0], sep='')

# We can find that the number of anomalies in MNIST is much lower than in MNIST-FASHION