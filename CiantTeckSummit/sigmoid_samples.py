import tensorflow as tf
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

# Mass of trainin
traing_X = [
    [1, 3.50,    1,  5],
    [1, 4.50,  1.5,  8],
    [1, 5.00,  1.6,  9],
    [1, 5.15,  1.7, 12],
    [1, 5.17,  1.8, 12],
    [1, 3.85,  2.1, 13],
    [1, 4.78,  2.5, 14],
    [1, 5.08,  2.3, 17],
    [1, 5.15, 2.12, 18],
    [1, 5.17,  3.1, 21]
]

traing_Y = [
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
]

# Params trainings
limt_epoch = 1000
step_show = 50
rate_learning = 0.01
num_samples = 10

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 1])

# variables tha will be calculeted by Tensorflow
a = tf.Variable(tf.zeros([4, 1]))

hypothesis = tf.sigmoid(tf.matmul(X, a))
error = tf.reduce_mean(-(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis)))
otimizador = tf.train.GradientDescentOptimizer(rate_learning).minimize(error) 

def sigmoid(x, derivative=False):
  return round(x*(1-x) if derivative else 1/(1+np.exp(-x)),2)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    merged = tf.summary.merge_all() # new
    writer = tf.summary.FileWriter("logs", sess.graph) # new
    sess.run(init)
    for Epoch in range(limt_epoch):
        result = sess.run([otimizador, error, hypothesis, a], {X: traing_X, Y: traing_Y})
        #print("a", sess.run(a))
        #print("error", sess.run([hypothesis, error],  {X: traing_X, Y: traing_Y}))
        #print("hypothesis", sess.run(hypothesis,  {X: traing_X, Y: traing_Y}))
        if ( Epoch+1) % step_show == 0:
            print("Epoch current:", '%04d' % (Epoch+1), "error=", "{:.9f}".format(result[1]))
        
    print("Finished")
    print("error found=", result[1], "\n", "a=", result[3])
    print("Sigmoid False=", sigmoid(np.matmul([1, 3.5, 1, 5], result[3])[0],False))
    print("Sigmoid True=", sigmoid(np.matmul([1, 5.17, 3.1, 21], result[3])[0],False))