import numpy as np
x = np.array([ 18,   23,   28,   33,   38,   43,   48,   53,  58,   63]).reshape(10,1)
y = np.array([871, 1132, 1042, 1356, 1488, 1638, 1569, 1754,1866, 1900]).reshape(10,1)

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)

import matplotlib.pyplot as plt
plt.scatter(x , y)
plt.show()

print(np.random.seed(0))
print(np.random.rand(2))

import tensorflow as tf

#Low level Tensorflow
# Formula : y = b0 + b1 * x0
b0 = tf.Variable(0.54)
b1 = tf.Variable(0.71)

erro = tf.losses.mean_squared_error(y, (b0 + b1 * x) )
otimizador = tf.train.GradientDescentOptimizer(learning_rate=0.001)
treinamento = otimizador.minimize(erro)
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for i in range(1000):
        session.run(treinamento)
        print(session.run([b0, b1, erro]))

    b0_final, b1_final = session.run([b0, b1])
#Modelo: y = b0 + b1 * x1
#Coeficiente
#b0
print(b0_final)
#b1
print(b1_final)
#Previsão Manual:
quarentaAnos = scaler_x.fit_transform( [[40]] )
previsao1 = b0_final + ( b1_final * quarentaAnos )
print(scaler_y.inverse_transform([previsao1]))

previsores = b0_final + (b1_final * x)
print(previsores)

resultado = y - previsores
print(abs(resultado))
print(abs(resultado).mean())

#Metricas via SkLearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean = mean_absolute_error(y, previsores)
mse = mean_squared_error(y, previsores)
print(mean, mse)

plt.plot(x , y, 'o')
plt.plot(x, previsores, '+')
plt.title('Regressao Linear - Sklearn')
plt.xlabel('Idade')
plt.ylabel('Salário')
plt.show()

y1 = scaler_y.transform(y)
previsao1 = scaler_y.transform(previsores)
print(previsao1)