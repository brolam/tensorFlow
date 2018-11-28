#https://archive.ics.uci.edu/ml/datasets.html

#1.Lendo a base
import pandas as pdb
base = pdb.read_csv('dados/census.csv')
#2.Definindo as variaves Numericas e Categóricas
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values
#3.Transformando as variáves Categóricas em Labels
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
labelEncoder_previsores = LabelEncoder()
#labels = labelEncoder_previsores.fit_transform(previsores[:,1])
previsores[:,1] = labelEncoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelEncoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelEncoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelEncoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelEncoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelEncoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelEncoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelEncoder_previsores.fit_transform(previsores[:,13])
#Eliminando a hieraquia entre as veriáves categórica com o OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[1,3,4,6,8,9,13])
previsores = oneHotEncoder.fit_transform(previsores).toarray()
#Aplicando o OneHotEnconde na variavel de classificção
labelEncoder_classe = LabelEncoder()
classe = labelEncoder_classe.fit_transform(classe)
#Escalonamento dos atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
