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


#plt.plot(house_size, house_price, "bx")
#plt.ylabel("Price")
#plt.xlabel("Size")
#plt.savefig('display.svg') 

