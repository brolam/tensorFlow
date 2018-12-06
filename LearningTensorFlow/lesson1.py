#http://learningtensorflow.com


# import tensorflow as tf
# x = tf.constant(35, name='x')
# y = tf.Variable(x + 5, name='y')

# print(y)

# model = tf.global_variables_initializer()

# with tf.Session() as session:
#     session.run(model)
#     print(session.run(y))


# import tensorflow as tf
# x = tf.constant([35, 40, 45], name='x')
# y = tf.Variable(x + 5, name='y')


# model = tf.global_variables_initializer()

# with tf.Session() as session:
# 	session.run(model)
# 	print(session.run(y))
	
# import tensorflow as tf

# x = tf.constant(35, name='x')
# print(x)
# y = tf.Variable(x + 5, name='y')

# with tf.Session() as session:
#     merged = tf.summary.merge_all()
#     writer = tf.summary.FileWriter("./tmp/basic", session.graph)
#     model =  tf.global_variables_initializer()
#     session.run(model)
#     print(session.run(y))

# import tensorflow as tf
    
    
# a = tf.constant([1, 2, 3], name='a')
# b = tf.constant(4, name='b')
# add_op = a + b

# with tf.Session() as session:
#     print(session.run(add_op))
    
    
# import tensorflow as tf

# a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
# b = tf.constant(100, name='b')
# add_op = a + b

# with tf.Session() as session:
#     print(session.run(add_op))
    
import tensorflow as tf    
    
a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
b = tf.constant([[100], [101]], name='b')
add_op = a + b

with tf.Session() as session:
    print(session.run(add_op))    
    