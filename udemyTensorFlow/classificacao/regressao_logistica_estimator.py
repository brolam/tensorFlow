import pandas as pd
base = pd.read_csv('/workspace/tensorFlow/udemyTensorFlow/classificacao/census.csv')
base.head()
base.income.unique()

def converte_classe(rotulo):
    if (rotulo ==  ' >50K'):
        return 1
    else:
        return 0

base.income = base.income.apply(converte_classe)

x = base.drop('income', axis=1)
y = base.income
type(y)
import matplotlib.pyplot as plt
x.age.hist()
plt.show()

import tensorflow as tf
idade = tf.feature_column.numeric_column('age')
idade_categoria = [tf.feature_column.bucketized_column(idade, boundaries=[20,30, 40, 50, 60, 70, 80, 90])]
x.columns
nomes_colunas_categoricas = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
colunas_categoricas = [tf.feature_column.categorical_column_with_vocabulary_list(key=c, vocabulary_list=x[c].unique()) for c in nomes_colunas_categoricas]
nomes_colunas_numericas = ['final-weight', 'education-num', 'capital-gain', 'capital-loos', 'hour-per-week']
colunas_numericas = [tf.feature_column.numeric_column(key=c) for c in nomes_colunas_numericas]
colunas = idade_categoria + colunas_categoricas + colunas_numericas
colunas
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x,y, test_size=0.30)
x_treinamento.shape
x_teste.shape

funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treinamento, y = y_treinamento, batch_size=32, num_epochs=None, shuffle=True)
classificador = tf.estimator.LinearClassifier(feature_columns=colunas)
classificador.train(input_fn=funcao_treinamento, steps=10000)

funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = x_teste, batch_size=32, shuffle=False)
previsoes = classificador.predict(input_fn=funcao_previsao)
list(previsoes)

previsao_final = []
for p in  classificador.predict(input_fn=funcao_previsao):
    previsao_final.append(p['class_ids'])
previsao_final

from sklearn.metrics import accuracy_score
taxa_acerto = accuracy_score(y_teste, previsao_final)
taxa_acerto