import pandas as pd
base = pd.read_csv('/workspace/tensorFlow/udemyTensorFlow/classificacao/census.csv')
base.head()
base.income.unique()
base.shape
x = base.iloc[:,0:14].values
y = base.iloc[:,14].values
x
y
from sklearn.preprocessing import LabelEncoder
label_enconde = LabelEncoder()
x[0]
for colun_encode_index in [1,3,5,6,7,8,9,13]:
    print(colun_encode_index)
    x[:,colun_encode_index] = label_enconde.fit_transform(x[:,colun_encode_index])

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x = scaler_x.fit_transform(x)

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x , y, test_size = 0.3)

x_treinamento.shape

from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression()
classificador.fit(x_treinamento, y_treinamento)

previsoes = classificador.predict(x_teste)
previsoes

from sklearn.metrics import accuracy_score
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_acerto