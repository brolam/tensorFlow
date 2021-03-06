import pandas as pd

base = pd.read_csv('/workspace/tensorFlow/udemyMLDS/RegresaoLinear/dados/plano-saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1:2].values

import matplotlib.pyplot as plt
from sklearn.svm import SVR
# kernel linear
""" from sklearn.svm import SVR
regressor_linear = SVR(kernel = 'linear')
regressor_linear.fit(X, y)

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor_linear.predict(X), color = 'red')
plt.show()
regressor_linear.score(X, y)
"""
# kernel poly
""" regressor_poly = SVR(kernel = 'poly', degree = 3)
regressor_poly.fit(X, y)

plt.scatter(X, y)
plt.plot(X, regressor_poly.predict(X), color = 'red')
plt.show()
regressor_poly.score(X, y)
 """
# kernel rbf
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

regressor_rbf = SVR(kernel = 'rbf')
regressor_rbf.fit(X, y)

plt.scatter(X, y)
plt.plot(X, regressor_rbf.predict(X), color = 'red')
plt.show()
regressor_rbf.score(X, y)

#previsao1 = scaler_y.inverse_transform(regressor_linear.predict(scaler_x.transform(40)))
#previsao2 = scaler_y.inverse_transform(regressor_poly.predict(scaler_x.transform(40)))
#previsao3 = scaler_y.inverse_transform(regressor_rbf.predict(scaler_x.transform(40)))

