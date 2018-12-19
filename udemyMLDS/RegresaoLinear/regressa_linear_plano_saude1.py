import pandas as pd
base = pd.read_csv('/workspace/tensorFlow/udemyMLDS/RegresaoLinear/dados/plano-saude.csv', encoding='utf-8')

x = base.iloc[:,0].values
y = base.iloc[:,1]

import numpy as np
correlacao = np.corrcoef(x, y)
print(correlacao)
print('X:', x)
x = x.reshape(-1,1)
print('X Reshaped:', x)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)
print('b1:', regressor.intercept_)
print('b0:', regressor.coef_)

import matplotlib.pyplot as plt
plt.scatter(x , y)
plt.plot(x, regressor.predict(x), color = 'red')
plt.title("Regresão Linear Simples")
plt.xlabel("Idade")
plt.ylabel("Custo")
plt.show()

previsao1 = regressor.predict(40)
previsao2 = regressor.intercept_ + regressor.coef_ * 40
print('Pevisões', previsao1, previsao2)

score = regressor.score(x, y)
print('Score:', score)