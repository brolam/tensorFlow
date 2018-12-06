# Imports
import network1
import numpy as np

net = network1.Network([3, 3, 1])


traing_X = np.array([
    ( 3.50,    1,  5),
    ( 4.50,  1.5,  8),
    ( 5.00,  1.6,  9),
    ( 5.15,  1.7, 12),
    ( 5.17,  1.8, 12),
    ( 3.85,  2.1, 13),
    ( 4.78,  2.5, 14),
    ( 5.08,  2.3, 17),
    ( 5.15, 2.12, 18),
    ( 5.17,  3.1, 21)
])

traing_Y = np.array([
     [0],
     [0],
     [0],
     [0],
     [0],
     [1],
     [1],
     [1],
     [1],
     [1]
])



training_data = zip(traing_X, traing_Y  )
training_data = list(training_data)
test_data = training_data

net.SGD(training_data, 100000, 5, 0.03, test_data)

print (np.argmax(net.feedforward((3.50,    1,  5))))
print (np.argmax(net.feedforward([5.17,  3.1, 21])))
#print (net.feedforward())
print('Biases', net.biases)
print('Weights', net.weights)
