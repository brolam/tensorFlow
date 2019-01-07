import pandas as pd
colunas_usadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']
base = pd.read_csv('/workspace/tensorFlow/udemyTensorFlow/regrecaoLinear/house-prices.csv', usecols=colunas_usadas)
base.head()

from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

print(colunas_usadas[1:])

base[colunas_usadas[1:]] = scaler_x.fit_transform(base[colunas_usadas[1:]])
base[['price']] = scaler_y.fit_transform(base[['price']])
base.head()
X = base.drop('price', axis = 1)
Y = base.price
X.head()
Y.head()
type(X)
type(Y)

import tensorflow as tf
colunas = [tf.feature_column.numeric_column(key=c) for c in colunas_usadas[1:]]
colunas[0]

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(X , Y, test_size = 0.3)
x_treinamento.shape
x_teste.shape
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treinamento, y=y_treinamento, num_epochs=None ,shuffle=True)
funcao_test = tf.estimator.inputs.pandas_input_fn(x = x_teste, y=y_teste, num_epochs=10000 ,shuffle=False)
regressor = tf.estimator.LinearRegressor(feature_columns=colunas)
regressor.train(input_fn=funcao_treinamento, steps=10000)
metricas_treinamento = regressor.evaluate(input_fn = funcao_treinamento, steps= 10000)
metricas_teste = regressor.evaluate(input_fn = funcao_test, steps= 10000)

metricas_treinamento
metricas_teste

funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = x_teste, shuffle=False)
previsoes = regressor.predict(input_fn=funcao_previsao)
list(previsoes)
valores_previsores = []
for p in regressor.predict(input_fn=funcao_previsao):
    valores_previsores.append(p['predictions'])

valores_previsores

import numpy as np
valores_previsores = np.asarray(valores_previsores).reshape(-1,1)
valores_previsores = scaler_y.inverse_transform(valores_previsores)
y_teste2 = scaler_y.inverse_transform(y_teste.values.reshape(-1,1))
y_teste2

#Metricas via SkLearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean = mean_absolute_error(y_teste2, valores_previsores  )
mse = mean_squared_error(y_teste2, valores_previsores)
print(mean, mse)
