import pandas as pd
base = pd.read_csv('/workspace/tensorFlow/udemyMLDS/RegresaoLinear/dados/house-prices.csv')

x = base.iloc[:,5:6].values
y = base.iloc[:,2].values

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_treinamento, y_treinamento)
score = regressor.score(x_treinamento, y_treinamento)
print('Score:', score)

import matplotlib.pylab as plt
plt.scatter(x_treinamento, y_treinamento)
plt.plot(x_treinamento, regressor.predict(x_treinamento), color = 'red')
plt.show()

print( 'Analise Manual' )
previsores  = regressor.predict(x_teste)
resultado = y_teste - previsores
print(abs(resultado))
print(abs(resultado).mean())
print( 'Analise Sklearn')
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_teste, previsores)
print(mae)
