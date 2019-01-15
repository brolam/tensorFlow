import pandas as pd
base = pd.read_csv('/workspace/tensorFlow/udemyTensorFlow/classificacao/census.csv')
base.head()
base.income.unique()

def converte_classe(rotulo):
    return 1 if ( rotulo == ' >50K' ) else 0

base.income = base.income.apply(converte_classe)

base.head
x = base.drop('income', axis=1)
x
y = base.income

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3)
x_treinamento.shape

base.columns

import tensorflow as tf

workclass = tf.feature_column.categorical_column_with_hash_bucket(key='workclass', hash_bucket_size=100)
education  = tf.feature_column.categorical_column_with_hash_bucket(key='education', hash_bucket_size=100)
marital_status = tf.feature_column.categorical_column_with_hash_bucket(key='marital-status', hash_bucket_size=100)
occupation = tf.feature_column.categorical_column_with_hash_bucket(key='occupation', hash_bucket_size=100)
relationship = tf.feature_column.categorical_column_with_hash_bucket(key='relationship', hash_bucket_size=100)
race  = tf.feature_column.categorical_column_with_hash_bucket(key='race', hash_bucket_size=100)
native_country  = tf.feature_column.categorical_column_with_hash_bucket(key='native-country', hash_bucket_size=100)

base['final-weight'].unique()
sex = tf.feature_column.categorical_column_with_vocabulary_list(key='sex', vocabulary_list=[' Male', ' Female'])
age = tf.feature_column.numeric_column(key='age')
final_weight = tf.feature_column.numeric_column(key=' final-weigh')
education_num = tf.feature_column.numeric_column(key='education-num')
capital_gain = tf.feature_column.numeric_column(key='capital-gain')
capital_loos = tf.feature_column.numeric_column(key='capital-loos')
hour_per_week = tf.feature_column.numeric_column(key='hour-per-week')

colunas = [age, workclass, final_weight, education, education_num,
marital_status, occupation, relationship, race, sex,
       capital_gain, capital_loos, hour_per_week, native_country]

colunas

funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x= x_treinamento, y=y_treinamento, batch_size=32, num_epochs= None, shuffle=True)
classificador = tf.estimator.DNNClassifier(hidden_units=[8, 8], feature_columns=colunas, n_classes=2)
classificador.train(input_fn=funcao_treinamento)
