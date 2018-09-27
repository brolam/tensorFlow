import os
import tensorflow as tf
import numpy as np
import math
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generation some house sizes between 1000 and 3500 (typical sq ft of house)
num_house = 160
num_seed = 42
np.random.seed(num_seed)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# Generate house prices from house size with a random noise added.
np.random.seed(num_seed)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

print(house_size)
print(house_price)

def normalize(array):
    return ( array - array.mean()) / array.std()

# define number of training samples, 0,7 = 70%. We can take the first 70% since the values are randominze     
num_train_samples = math.floor(num_house * 0.7)
print('Amount Samples: {0} / {1}'.format(num_train_samples, num_house))

#define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_house_price = np.asanyarray(house_price[:num_train_samples])
print('Amount Samples Traing Size:{0} / Price:{1}'.format(train_house_size.size, train_house_price.size))

train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

print("Traing Size Normalize: ",train_house_size_norm)
print("Traing Price Normalize", train_house_price_norm)

# Set up the TensorFlow placeholders that get updated as we descend the gradient
tf_house_size = tf.placeholder("float", name="house_size")
tf_house_price = tf.placeholder("float", name="house_price")

# Define the variables holding the size_factor and price we set during training.
# We initialize them to some random values based on the normal distribution.
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(),"")

#plt.plot(house_size, house_price, "bx")
#plt.ylabel("Price")
#plt.xlabel("Size")
#plt.savefig('display.svg') 

