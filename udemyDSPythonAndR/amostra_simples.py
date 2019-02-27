import pandas as pd
import numpy as np
base = pd.read_csv('/workspace/tensorFlow/udemyDSPythonAndR/Estatistica1/infert.csv')
base.shape
amostra = np.random.choice(a=[0,1], size = 150, replace=True, p=[0.5,0.5])
amostra
len(amostra)
len(amostra[amostra == 0])
len(amostra[amostra == 1])




