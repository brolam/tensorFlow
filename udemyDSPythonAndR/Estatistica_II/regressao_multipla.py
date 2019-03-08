import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

base =  pd.read_csv('/workspace/tensorFlow/udemyDSPythonAndR/Estatistica_II/mt_cars.csv')
base = base.drop(['Unnamed: 0'], axis = 1)

base.iloc[0:2, 1:2]

X = base.iloc[:, 2].values
X
y = base.iloc[:, 0]
y
correlacao = np.corrcoef(X, y)
correlacao
X = X.reshape(-1, 1)
modelo = LinearRegression()
modelo.fit(X, y)
modelo.intercept_
modelo.coef_
modelo.score(X, y)

previsoes = modelo.predict(X)
previsoes

plt.scatter(X, y)
plt.plot(X, previsoes, color= 'red')

X1 = base.iloc[:, 1:4].values
modelo2 = LinearRegression()
modelo2.fit(X1, y)
previsoes2 = modelo2.predict(X1)
previsoes2

fig = plt.subplot()
ax = fig.scatter(X, previsoes2, color= 'green')
fig.plot(X, previsoes2, color= 'green')
plt.show()


