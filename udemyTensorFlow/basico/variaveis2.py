import tensorflow as tf
vetor = tf.constant([5, 10, 15], name='vetor')
type(vetor)
print(vetor)
soma = tf.Variable(vetor + 5, name='soma')
type(soma)
print(soma)
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    s = session.run(soma)
print(s)

valor = tf.Variable(0, name='valor')
init2 = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init2)
    for i in range(5):
        valor = valor + 1
        print(session.run(valor))