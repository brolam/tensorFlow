import pandas as pd
base = pd.read_csv('/workspace/tensorFlow/udemyTensorFlow/regrecaoLinear/house-prices.csv')
base.head()
base.count()
x = base.iloc[:, 5].values
x = x.reshape(-1,1)
x
y = base.iloc[:, 2:3].values
y
x.shape
y.shape

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)

import matplotlib.pyplot as plt
plt.scatter(x , y)
plt.show()

import numpy as np
print(np.random.seed(0))
print(np.random.rand(2))

import tensorflow as tf
b0 = tf.Variable(0.54)
b1 = tf.Variable(0.71)
bathsize = 32
xph = tf.placeholder(tf.float32, [bathsize, 1])
yph = tf.placeholder(tf.float32, [bathsize, 1])
y_modelo = b0 + (b1 * xph)
erro = tf.losses.mean_squared_error(y_modelo, yph)
otimizador = tf.train.GradientDescentOptimizer(0.001)
treinamento = otimizador.minimize(erro)
init =  tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for i in range(10000):
        indexs = np.random.randint(len(x), size=bathsize) 
        feed = {xph: x[indexs], yph:y[indexs]}
        session.run(treinamento, feed_dict=feed)
    b0_final, b1_final = session.run([b0, b1])
        
print (b0_final)
print (b1_final)

previsores = b0_final + b1_final * x

plt.plot(x , y, 'o')
plt.plot(x, previsores, '+')
plt.title('Regressao Linear - Sklearn')
plt.xlabel('Idade')
plt.ylabel('Salário')
plt.show()

y1=scaler_y.inverse_transform(y)
previsores1=scaler_y.inverse_transform(previsores)
y1
previsores1

#Metricas via SkLearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean = mean_absolute_error(y1, previsores1)
mse = mean_squared_error(y1, previsores1)
print(mean, mse)
