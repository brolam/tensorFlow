from scipy.stats import t

# Media de salários dos cientistas de dados = R$ 75,00 por hora 
# Amostra com 9 funcionários e desvio padrão = 10

# Quak a probabilidade de selecionar um cientista de dados e o salário ser maior que R$ 80 por hora.
t.cdf(1.5, 8)
t.sf(1.5, 8)

t.cdf(1.5, 8)  + t.sf(1.5, 8)