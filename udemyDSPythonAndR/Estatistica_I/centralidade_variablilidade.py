import numpy as np
from scipy import  stats
jogadores = [40000, 18000, 12000, 250000, 30000, 140000, 300000, 40000, 800000]
np.mean(jogadores)
np.median(jogadores)
quartis = np.quantile(jogadores, [0, 0.25, 0.5, 0.75, 1])
quartis
np.std(jogadores)
stats.describe(jogadores)

np.median([22,10,12,14,13,15])