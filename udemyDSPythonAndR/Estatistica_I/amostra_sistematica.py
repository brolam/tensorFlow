import numpy as np
import pandas as pd

np.random.choice(a = [0, 1], size = 50, replace = True, p = [0.5, 0.5, 0.2])
from math import ceil

populacao = 150
amostra = 15
k = ceil(populacao / amostra)
r = np.random.randint(low=1, high=k + 1, size=1)
r
acumulador = r[0]
sorteados = []
for i in range(amostra):
    sorteados.append(acumulador)
    acumulador += k
sorteados

iris = pd.read_csv('/workspace/tensorFlow/udemyDSPythonAndR/Estatistica1/iris.csv')
base_final = iris.loc[sorteados]
base_final
