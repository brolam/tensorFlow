import os
import tensorflow as tf
import numpy as np
import math
import matplotlib


#matplotlib.use("agg")
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generation some house sizes between 1000 and 3500 (typical sq ft of house)
num_house = 160
np.random.seed(60)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# Generate house prices from house size with a random noise added.
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

#print(house_size)
#print(house_price)

plt.plot(house_size, house_price, "bx")
plt.ylabel("Price")
plt.xlabel("Size")
plt.savefig('display.svg') 
#plt.show()

