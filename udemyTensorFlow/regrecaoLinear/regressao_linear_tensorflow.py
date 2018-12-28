import numpy as np
x = np.array([ 18,   23,   28,   33,   38,   43,   48,   53,  58,   63]).reshape(10,1)
y = np.array([871, 1132, 1042, 1356, 1488, 1638, 1569, 1754,1866, 1900]).reshape(10,1)
print(x, y)

import matplotlib.pyplot as plt
plt.scatter(x , y)
plt.show()

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x = scaler_x.fit(x)
y = scaler_y.fit(y)

plt.scatter(x , y)
plt.show()

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)

#Modelo: y = b0 + b1 * x1
#Coeficiente
#b0
print(regressor.intercept_)
#b1
print(regressor.coef_)
#Previsão Manual:
quarentaAnos = 40
previsao1 = regressor.intercept_ + ( regressor.coef_ * quarentaAnos )
print(previsao1)
#Previsão via o previsor
print( regressor.predict(40))
previsores = regressor.predict(x)
print(previsores)

resultado = y - previsores
print(abs(resultado))
print(abs(resultado).mean())

#Metricas via SkLearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean = mean_absolute_error(y, previsores)
mse = mean_squared_error(y, previsores)
print(mean, mse)

plt.plot(x , y, 'o')
plt.plot(x, previsores, '+')
plt.title('Regressao Linear - Sklearn')
plt.xlabel('Idade')
plt.ylabel('Salário')
plt.show()