import tensorflow as tf
import numpy as np

x = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

x
y = np.array([
    [0.0],
    [0.0],
    [0.0],
    [1.0]
])

y
neuronios_entradas = 2
neuronios_oculta = 3
neuronios_saida = 1

w = {
    'oculta': tf.Variable(tf.random_normal([neuronios_entradas, neuronios_oculta]), name='w_oculta'),
    'saida': tf.Variable(tf.random_normal([neuronios_oculta, neuronios_saida]), name='w_saida')
    }

b = {
    'oculta': tf.Variable(tf.random_normal([neuronios_oculta]), name='b_oculta'),
    'saida': tf.Variable(tf.random_normal([neuronios_saida]), name='b_saida')
    }


w['oculta']
w['saida']

xph = tf.placeholder(tf.float32, [4, neuronios_entradas], name='xph')
yph = tf.placeholder(tf.float32, [4, neuronios_saida], name='yph')

camada_oculta = tf.add(tf.matmul(xph, w['oculta']), b['oculta'])
camada_oculta_ativacao = tf.sigmoid(camada_oculta)
camada_saida = tf.add(tf.matmul(camada_oculta_ativacao, w['saida']), b['saida'])
camada_saida_ativacao = tf.sigmoid(camada_saida)
erro = tf.losses.mean_squared_error(yph, camada_saida_ativacao)
otimizador = tf.train.GradientDescentOptimizer(learning_rate=0.03).minimize(erro)
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    #print(session.run(camada_saida_ativacao, feed_dict={xph:x, yph:y}))
    for epocas in range(10000):
        erro_medio = 0 
        _, custo = session.run([otimizador, erro], feed_dict= {xph:x, yph:y})
        if epocas %  200 == 0:
            erro_medio += custo / 4
            print(erro_medio)
    print(session.run(camada_saida_ativacao, feed_dict={xph:x, yph:y}))

