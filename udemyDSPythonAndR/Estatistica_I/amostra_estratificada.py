import pandas as pd
from sklearn.model_selection import train_test_split

iris = pd.read_csv('/workspace/tensorFlow/udemyDSPythonAndR/Estatistica1/iris.csv')
iris.head
iris['class'].value_counts()

x, _, y, _ = train_test_split(iris.iloc[:,0:4], iris.iloc[:, 4], test_size = 0.5, stratify= iris.iloc[:, 4])
len(x)
len(y)
y.value_counts()

infert = pd.read_csv('/workspace/tensorFlow/udemyDSPythonAndR/Estatistica1/infert.csv')
infert.head
infert['education'].value_counts()
#6-11yrs    120
#12+ yrs    116
#0-5yrs      12
total_education = len(infert['education'])
total_education
( 120 / total_education ) * 100 # 6-11yrs
( 116 / total_education) #12+ yrs
(  12 / total_education) #0-5yrs

x, _, y, _ = train_test_split(infert.iloc[:,2:29], infert.iloc[:, 1], test_size = 0.6, stratify= infert.iloc[:, 1])
y.value_counts()