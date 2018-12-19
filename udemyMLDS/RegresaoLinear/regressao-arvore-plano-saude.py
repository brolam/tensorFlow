import pandas as pd

base = pd.read_csv('/workspace/tensorFlow/udemyMLDS/RegresaoLinear/dados/plano-saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)
score = regressor.score(X, y)

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title('Regressão com redes neurais')
plt.xlabel('Idade')
plt.ylabel('Custo')
#plt.show()

import numpy as np
X_teste = np.arange(min(X), max(X), 0.1)
X_teste = X_teste.reshape(-1, 1)
plt.scatter(X, y)
plt.plot(X_teste, regressor.predict(X_teste), color = 'red')
plt.title('Regressão com redes neurais')
plt.xlabel('Idade')
plt.ylabel('Custo')
plt.show()