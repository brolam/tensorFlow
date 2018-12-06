import tensorflow as tf
import numpy as np
import os

# Disable compiler warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Disable warnings.
tf.logging.set_verbosity(tf.logging.INFO)

# One feature.
features_x = [tf.contrib.layers.real_valued_column("x", dimension=1)]
features_y = [tf.contrib.layers.real_valued_column("y", dimension=1)]

# Use linear regressor with 1 feature.
estimator_x = tf.contrib.learn.LinearRegressor(feature_columns=features_x)
estimator_y = tf.contrib.learn.LinearRegressor(feature_columns=features_y)

# Two NumPy arrays with float32 type.
# ... We want to get a function to go from first to second array.
# x_train = np.array([1., 2., 3.,  5.,  2.,  6.,  7.], dtype=np.float32)
# y_train = np.array([8., 9., 10., 11., 12., 14., 16.], dtype=np.float32)
x_train = np.asanyarray([ 2.00, 4.50, 6.00,  8.90,  10.00, 12.80,  14.00, 16.50, 18.00, 20.50, 22.00, 24.10], dtype=np.float32)
y_train = np.asanyarray([ 1.00, 2.00, 3.00,  4.00,   5.00,  6.00,   7.00,  8.00,  9.00, 10.00, 11.00, 12.00], dtype=np.float32)


# Use special method to get an input_fn based on NumPy arrays.
input_fn_x = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train,
                                              batch_size=2,
                                              num_epochs=1000)
                                              
input_fn_y = tf.contrib.learn.io.numpy_input_fn({"y":y_train}, x_train,
                                              batch_size=2,
                                              num_epochs=1000)

# Fit the model to our training data.
estimator_x.fit(input_fn=input_fn_x, steps=10000)
estimator_y.fit(input_fn=input_fn_y, steps=10000)

# Predict scores for the x feature column.
res1 = estimator_x.predict_scores({"x":np.asanyarray([24.10],  dtype=np.float32)})
res2 = estimator_y.predict_scores({"y":np.asanyarray([12.00],  dtype=np.float32)})

# Convert generator to list and print it.
test1 = list(res1)
test2 = list(res2)
print("predict_scoresX", test1)
print("predict_scoresY", test2)