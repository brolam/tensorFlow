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
w = tf.Variable(tf.zeros([2,1], dtype= tf.float64))
type(w)
w
def step(x):
    return tf.cast(tf.to_float(tf.math.greater_equal(x, 1)), tf.float64)
init = tf.global_variables_initializer()
camada_saida = tf.matmul(x , w)
camada_saida_ativacao = step(camada_saida)
erro = tf.subtract(y, camada_saida_ativacao)
delta = tf.matmul(x, erro, transpose_a=True)
treinamento = tf.assign(w, tf.add(w, tf.multiply(delta, 0.1)))

with tf.Session() as session:
    session.run(init)
    epoca = 0
    for i in range(15):
        epoca+=1
        erro_total, _ = session.run([erro, treinamento]) 
        erro_soma = tf.reduce_sum(erro_total)
        print('Epoca:', epoca, ' Error:', session.run(erro_soma), ' W:', w.eval())
        if erro_soma.eval() == 0:
            break
        w_final = w.eval()

camada_saida_teste = tf.matmul(x, w_final)
camada_saida_ativacao_teste = step(camada_saida_teste)
with tf.Session() as session:
    session.run(init)
    print(session.run(camada_saida_ativacao_teste))