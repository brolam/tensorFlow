import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

base_cars = pd.read_csv('/workspace/tensorFlow/udemyDSPythonAndR/Estatistica_II/cars.csv')
base_cars = base_cars.drop(['Unnamed: 0'], axis = 1)
X = base_cars.iloc[:, 1].values.reshape(-1, 1)
y = base_cars.iloc[:, 0].values
X
y
corelacao = np.corrcoef(X.reshape(1,-1), y)
corelacao

modelo = LinearRegression()
modelo.fit(X, y)

modelo.intercept_
modelo.coef_

plt.scatter(X, y)
plt.plot(X, modelo.predict(X), color='red')
plt.show()

#Distancia 22 pés 
modelo.intercept_ + modelo.coef_ * 22

modelo.predict([[22]])

modelo._residues

