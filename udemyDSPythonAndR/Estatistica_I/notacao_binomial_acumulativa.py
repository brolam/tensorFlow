from scipy.stats import binom

binom.pmf(8, 10, 0.5)

# Jogar uma moeda 5 vezes, qual a probabilidade de dar 3 vezes?
prob = binom.pmf(3, 5, 0.5)
prob

# Passar por 4 sinais de 4 tempos, qual a probabilidade de pegar sinal verde;
# nenhuma, 1, 2, 3 ou 4 vezes seguidas?
binom.pmf(0, 4, 0.25)
binom.pmf(1, 4, 0.25)
binom.pmf(2, 4, 0.25)
binom.pmf(3, 4, 0.25)
binom.pmf(4, 4, 0.25)

# E se forem sinais de dois tempos?
binom.pmf(4, 4, 0.5)

# Propabilidade acumulativa
binom.cdf(4, 4, 0.25)

# Concurso em 12 questoes, qual a probabilidade de acertar 7 questoes considerando 
# que cada questão tem 4 alternativas?
binom.pmf(7, 12, 0.25)
