import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
base = pd.read_csv('dados/Restaurante_2_Estrelas.csv', sep=";")
previsores = base.iloc[:, 0:3].values
#print(previsores)
classe = base.iloc[:, 3].values
#print(classe)

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.40, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)
#print(previsoes)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
precisao = accuracy_score(classe_teste, previsoes)
print('Precisão:', precisao)
matriz = confusion_matrix(classe_teste, previsoes)
print('Confusion Matrix:')
print(matriz)
print('Classes:', classificador.classes_)
print('Classes Count:', classificador.class_count_)
print('Classes Prioridade:', classificador.class_prior_)
predict = classificador.predict([
    [-20,3,0], #0
    [-20,4,6], #1
    [-19,1,0], #0
    [-25,2,8], #1
    [-25,3,0], #0
]) 
print('Previsão:', predict)

print('Relatório')
print(classification_report(classe_teste, previsoes, target_names=classificador.classes_))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(matriz, classes=class_names,
#                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(matriz, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')

#plt.show()