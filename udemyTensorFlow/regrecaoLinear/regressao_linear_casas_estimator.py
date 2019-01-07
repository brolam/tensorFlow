import pandas as pd
base = pd.read_csv('/workspace/tensorFlow/udemyTensorFlow/regrecaoLinear/house-prices.csv')

base.head()
base.shape

x = base.iloc[:, 5:6]
y = base.iloc[:, 2:3]

x.shape
y.shape

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)
import tensorflow as tf
columas = [tf.feature_column.numeric_column('x', shape = [1])]
regressor = tf.estimator.LinearRegressor(feature_columns=columas)

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x , y, test_size = 0.3)
funcao_treinamento = tf.estimator.inputs.numpy_input_fn({'x':x_treinamento}, y_treinamento, batch_size=32, num_epochs=None, shuffle=True)
funcao_teste = tf.estimator.inputs.numpy_input_fn({'x':x_teste}, y_teste, batch_size=32, num_epochs= 1000, shuffle=False)
regressor.train(input_fn = funcao_treinamento, steps=10000)
metricas_treinamento = regressor.evaluate(input_fn=funcao_treinamento, steps=10000)
metricas_teste = regressor.evaluate(input_fn=funcao_teste, steps=100000)

metricas_teste
metricas_treinamento

import numpy as np
novas_cadas = np.array([[800], [900], [1000]])
novas_cadas = scaler_x.transform(novas_cadas)
novas_cadas
funcao_previsao = tf.estimator.inputs.numpy_input_fn({'x':novas_cadas}, shuffle=False)
previsores = regressor.predict(input_fn = funcao_previsao)
list(previsores)
for p in regressor.predict(input_fn = funcao_previsao):
    print(p['predictions'])