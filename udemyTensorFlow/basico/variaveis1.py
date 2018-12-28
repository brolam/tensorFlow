import tensorflow as tf
valor1 = tf.constant(15, name='valor1')
soma = tf.Variable(valor1 + 5, name='valor1')
type(soma)
print(soma)
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    s = session.run(soma)
print(s)