import tensorflow as tf
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

# Params trainings
rate_learning = 0.00001
limit_epoch = 2000
steps_show = 50

# Mass of trainigs (price vs fuel)
train_x = np.asanyarray([ 2.00, 4.50, 6.00,  8.90,  10.00, 12.80,  14.00, 16.50, 18.00, 20.50, 22.00, 24.10])
train_y = np.asanyarray([ 1.00, 2.00, 3.00,  4.00,   5.00,  6.00,   7.00,  8.00,  9.00, 10.00, 11.00, 12.00])
num_train_samples = math.floor(len(train_x) * 0.70)

print("x train", len(train_x))
print("y train", len(train_y))
print("train samples", num_train_samples)

# placeholders
X = tf.placeholder(tf.float32, name= "X")
Y = tf.placeholder(tf.float32, name= "Y")

# variables tha will be calculeted by Tensorflow
a = tf.Variable(0.0, name="a")
b = tf.Variable(0.0, name="b")

# Initial Analysis - Plot Mass of trainigs (price vs fuel)
plt.plot(train_x, train_y, "bx")  # bx = blue x
plt.xlabel("Fuel")
plt.ylabel("Price")
plt.axis('equal')
plt.savefig('display_price_vs_fuel_initial_analysis.svg') 

#hypothesis = tf.add(tf.multiply(a, X), b)
hypothesis =  X*a+b
error = tf.reduce_sum(tf.pow(Y - hypothesis, 2)) / num_train_samples
otimizador = tf.train.GradientDescentOptimizer(rate_learning).minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    merged = tf.summary.merge_all() # new
    writer = tf.summary.FileWriter("logs", sess.graph) 
    sess.run(init)
    
    for epoch in range(limit_epoch):
        for(x, y) in zip( train_x, train_y):
            sess.run(otimizador, feed_dict={X: x, Y: y})
        if ( epoch+1 ) % steps_show == 0:
            c = sess.run(error, feed_dict={X: train_x, Y: train_y})
            print("Current Epoch" , '%04d' % (epoch+1), "Error=" , "{:.9f}".format(c), "a=", sess.run(a), "b=", sess.run(b))
        
    print("Finished Exectution")
    error_final = sess.run(error, feed_dict={X: train_x, Y: train_y})
    print("Error final found=", error_final, "a=", sess.run(a), "b=", sess.run(b))
    
    trained_x = np.zeros(len(train_x))
    trained_y = np.zeros(len(train_y))
    for index, (x, y) in enumerate(zip( train_x, train_y)):
        trained_x[index] = (y / sess.run(a)) - sess.run(b)
        trained_y[index] = (sess.run(a) * x) + sess.run(b)
        print(index, x, y)
        print(index, round((trained_x[index]), 2), round(trained_y[index], 2))
        
    # Plot the graph
    plt.ylabel("Y")
    plt.xlabel("X (sq.ft)")
    plt.plot(train_x, train_y, 'go', label='Training data')
    plt.plot(trained_x, trained_y)
    plt.plot(trained_x, trained_y, "bx")
    plt.savefig('display_price_vs_fuel_learning.svg')  

