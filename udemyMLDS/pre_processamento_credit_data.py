import pandas as pd
base = pd.read_csv('dados/credit-data.csv')
#base.describe()
#apagar coluna
#base.drop('age', 1, inplace=True)
#pagar somente os registros com problema
#base.drop(base[base.age < 0].index, inplace=True)
#print(base.loc[base['age'] < 0 ])
#corrigindo as idades negativas com a média,
#base['age'].mean()
#base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92
#base.loc[pd.isnull(base['age']), 'age'] = 40.92
#base['age'].mean()
#corrigindo valores não informados
# pip install -U scikit-learn
#pd.isnull(base['age'])
#base.loc[pd.isnull(base['age'])]
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:,4].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:,0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])
#padronização das escalas entre as colunas
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:29:01 2017

@author: Jones
"""
import pandas as pd
base = pd.read_csv('credit_data.csv')
base.describe()
base.loc[base['age'] < 0]
# apagar a coluna
base.drop('age', 1, inplace=True)
# apagar somente os registros com problema
base.drop(base[base.age < 0].index, inplace=True)
# preencher os valores manualmente
# preencher os valores com a média
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92
        
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)





