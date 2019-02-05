import pandas as pd
base = pd.read_csv('/workspace/tensorFlow/udemyTensorFlow/RNR/petr4.csv')
base.head()
base = base.dropna()
base = base.iloc[:,1].values
import matplotlib.pyplot as plt
plt.plot(base)
plt.show()
periodos = 30
previsao_futura = 1

x = base[0:(len(base) - (len(base) % periodos))]