from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
base = pd.read_csv('dados/risco_credito.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
# Transformando valoes textos em 0 ou 1 com
labelEncoder = LabelEncoder()
previsores[:, 0] = labelEncoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelEncoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelEncoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelEncoder.fit_transform(previsores[:, 3])
classificador = GaussianNB()
classificador.fit(previsores, classe)
# história: boa, dívida: alta, garantias: nenhuma, renda > 35
resultado = classificador.predict([[0, 0, 1, 2]])
print('# história: boa, dívida: alta, garantias: nenhuma, renda > 35')
print(resultado)
# história: ruin, dívida: alta, garantias: adequada, renda < 15
print('# história: boa, dívida: alta, garantias: nenhuma, renda > 35')
print('# história: ruin, dívida: alta, garantias: adequada, renda < 15')
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])
print(reslsultado)
print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)
