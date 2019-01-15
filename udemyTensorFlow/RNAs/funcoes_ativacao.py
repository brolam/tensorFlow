import numpy as np

# Somente para problemas linear, problema linearmente separaveis
def stepFunction(soma):
    if ( soma >= 1 ):
        return 1
    return 0
# Somente para classificação binária Mais próximo de zero ou mais próximo de um
def sigmoidFunction(soma):
    print(np.exp(-soma))
    return 1 / (1 + np.exp(-soma))

#Tangente hiperbólica: Também para classifição zero ou um
def tahnFunction(soma):
    print( np.exp(soma), " - ", np.exp(-soma), "/", np.exp(soma), np.exp(-soma)   )
    return ( np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))
# Utilizadas nas RNs, principalmente nas camadas internas 
def relu(soma):
    if soma >= 0:
        return soma
    return 0
#Usada para regressão
def linearFunction(soma):
    return soma
#Usada para classificação quando só tem duas classes;
def softMax(x):
    ex = np.exp(x)
    return ex / ex.sum()

teste = stepFunction(-1)
teste
teste = sigmoidFunction(0.358)
teste
teste = tahnFunction(0.358)
teste
teste = softMax([7.0, 2.0, 1.3])
print(sigmoidFunction(2.1), tahnFunction(2.1), relu(2.1), linearFunction(2.1))