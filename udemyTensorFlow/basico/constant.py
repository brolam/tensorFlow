import tensorflow as tf
valor1 = tf.constant(2)
valor2 = tf.constant(3)
type(valor1) 
print(valor1, valor2)
soma = valor1 + valor2
type(soma)
print(soma)
with tf.Session() as session:
    s = session.run(soma)
print(s) 