import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

base = pd.read_csv('/workspace/tensorFlow/udemyDSPythonAndR/Estatistica_II/Eleicao.csv', sep=';')
base.head
plt.scatter(base.DESPESAS, base.SITUACAO)
plt.show()

base.describe()

np.corrcoef(base.DESPESAS, base.SITUACAO)

X = base.iloc[:, 2].values
X = X[:, np.newaxis]
X
y = base.iloc[:, 1].values
modelo = LogisticRegression()
modelo.fit(X, y)
modelo.coef_
modelo.intercept_

plt.scatter(X, y)
X_teste = np.linspace(10, 3000, 100)
X_teste

""" def model(x):
    return 1 / (1 + np.exp(-x))

r = model(X_teste * modelo.coef_ + modelo.intercept_).ravel()
 """

base_previsoes = pd.read_csv('/workspace/tensorFlow/udemyDSPythonAndR/Estatistica_II/NovosCandidatos.csv', sep=";")
base_previsoes

despesas = base_previsoes.iloc[:, 1].values.reshape(-1,1)

previsoes_teste = modelo.predict_proba(despesas)
np.transpose(previsoes_teste, axes=(0))
previsoes_teste[:,0]

plt.scatter(despesas.reshape(-1), previsoes_teste[:,1])
plt.show()

from sklearn.linear_model import LinearRegression

X2 = np.array([15,18,20,25,30,44])
X2
Y2 = np.array([240,255,270,283,300,310])
Y2

np.corrcoef(X2, Y2)
