from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(categorical_features=[0])
y = y.reshape(-1,1)
y = onehot.fit_transform(y).toarray()
y

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3)
x_treinamento.shape

import tensorflow as tf
import numpy as np

neuronios_entradas = x.shape[1]
neuronios_oculta = int(np.ceil( (x.shape[1] + y.shape[1] ) / 2 ))
neuronios_oculta

neuronios_saida = y.shape[1]
neuronios_saida

w = {
    'oculta': tf.Variable(tf.random_normal([neuronios_entradas, neuronios_oculta])),
    'saida': tf.Variable(tf.random_normal([neuronios_oculta, neuronios_saida]))
    }

b = {
    'oculta': tf.Variable(tf.random_normal([neuronios_oculta])),
    'saida': tf.Variable(tf.random_normal([neuronios_saida]))
    }

xph = tf.placeholder(tf.float32,[None, neuronios_entradas], name='xph')
yph = tf.placeholder(tf.float32,[None, neuronios_saida], name='yph')

def modelo(x , w, bias):
    camanda_oculta = tf.add(tf.matmul(x, w['oculta']), bias['oculta'])
    camanda_oculta_ativacao = tf.nn.relu(camanda_oculta)
    camanda_saida = tf.add( tf.matmul(camanda_oculta_ativacao, w['saida']), b['saida'])
    return camanda_saida

modelo = modelo(xph, w, b)
erro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=modelo, labels=yph))
otimizador = tf.train.AdadeltaOptimizer(learning_rate=0.03).minimize(erro)

batch_size = 8
batch_total = int(len(x_treinamento) / batch_size)
batch_total
x_batches = np.array_split(x_treinamento, batch_total)
x_batches

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for epoca in range(10000):
        erro_medio = 0.0
        batch_total = int(len(x_treinamento) / batch_size)
        x_batches = np.array_split(x_treinamento, batch_total)
        y_batches = np.array_split(y_treinamento, batch_total)
        for i in range(batch_total):
            x_batch, y_batch = x_batches[i], y_batches[i]
            _, custo = session.run([otimizador, erro], feed_dict={xph: x_batch, yph: y_batch})
            erro_medio += custo / batch_total
            if epoca % 500 == 0:
                print('Epoca: ', epoca + 1, ' Erro: ', erro_medio )
        w_final, b_final = session.run([w, b])

print(w_final , b_final)


#Previsão:
previsores_testes = modelo(xph, w_final, b_final)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    r1 = session.run(previsores_testes, feed_dict={xph: x_teste})
    r2 = session.run(tf.nn.softmax(r1))
    r3 = session.run(tf.argmax(r2, 1))

r1
r2
r3

y_teste2 = np.argmax(y_teste, 1)

y_teste2

from sklearn.metrics import accuracy_score
taxa_acerto = accuracy_score(y_teste2, r3)
taxa_acerto







