# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IBSR import IBSRegressor
from SGDR import OnlineBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3


from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from river import datasets


def streamData(DATASET_PATH):
    dataset = pd.read_csv(DATASET_PATH)#, header=0, index_col=0)
    #dataset.drop(columns='azul', inplace=True) # remove labels não numericos
    values = dataset.values
    # ensure all data is float >>> check if needed
    # values = values.astype('float32')
    return values
    #features = values[:, :-1]
    #labels = values[:,-1].reshape(-1, 1)


def streamDataArtificial():
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target

    #X, y = make_friedman3(10000)
    # Convertendo para DataFrame
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    # Combinando os dados
    df = pd.concat([df_X, df_y], axis=1)
    # Removendo os cabeçalhos
    df.columns = range(df.shape[1])
    values = df.values
    # ensure all data is float >>> check if needed
    # values = values.astype('float32')
    return values
    #features = values[:, :-1]
    #labels = values[:,-1].reshape(-1, 1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DATASET_PATH = 'housing.csv'

    #ds = streamData(DATASET_PATH)
    ds = streamDataArtificial()

    ibs =IBSRegressor()  # cria um modelo base
    ibs.partial_fit_Ens(ds[0,:-1],ds[0,-1],0) # fit(ds[0,:-1],ds[0,-1])  # treina com o primeiro exemplo
    ds = np.delete(ds, 0, axis=0)  # remove o exemplo usado

    performance_scores = [0]
    denominador = 1
    predictions = []
    prequential = []
    onlineWMAPE=[]
    alpha = 0.999  # prequential forgetting parameter
    cont = 1 # conta exemplos processados para calculo do MAE
    erro = 0
    ys = 0

    llist = []

    # simulando um fluxo de dados, testa e treina
    for row in ds:

        erroNominador = 0

        ##### testa ###
        #predictions.append(ibs.predict(row[:-1]))  # lista das predições
        #aux = ibs.predictTest(row[:-1]) #ibs.predict(row[:-1], row[-1]) # predicao - funcao que calcula weights  #
        aux = ibs.predictAndUpdate(row[:-1],row[-1])
        predictions.append(aux)
        erro += abs(aux - row[-1])  # erro da instancia atual - MAE

        # prequential
        #erro += abs(ibs.predict(row[:-1]) - row[-1])  # erro da instancia atual - MAE
        ys += abs(row[-1]) # soma iterativa dos ys

        ## P(20000) = al20000 ...  al^3 x e1/al^3 + al^2 x e2/al^2 + al^1 x e3/al^1 + al^0 x e4/al^0

        #errmae = erro/cont
        #mae = mean_absolute_error(actual, pred)
        #wmape = mae * len(pred)*100/sum(abs(actual))
        errmae = erro/ys  # online wmape
        onlineWMAPE.append(errmae)

        array = np.array(performance_scores) * alpha
        erroNominador = errmae + np.sum(array)
        prequential.append(erroNominador/denominador)
        denominador += alpha

        # erro wmape online
        performance_scores = list(array)
        performance_scores.append(errmae) # erro
        cont += 1

        #calculo para adição de regressores
        inner_term = math.exp(onlineWMAPE[-1])#/math.exp(0.2*onlineWMAPE[-1]) # math.exp(prequential[-1]) # /10
        #inner_term = math.exp(prequential[-1])#/math.exp(0.5*prequential[-1]) # math.exp(prequential[-1]) # /10
        L = math.ceil(inner_term) - 1
        #L = math.floor(onlineWMAPE[-1]*10)-1
        print(f"erro {onlineWMAPE[-1]} L {L}")
        #if L > 20:
        #    L = 20

        #print(ibs.get_n_estimators()) # imprime tamanho do ensemble para teste
        llist.append(ibs.get_n_estimators())

        ### treina ####
        # atualiza online - Poisson
        if cont > 50:
            ibs.partial_fit_Ens(row[:-1], row[-1], aux, L)
        else:
            ibs.partial_fit_Ens_old(row[:-1], row[-1],aux)


        # se erro > theta
        #adiciona n regressores a depender do erro - um a um ?

        # output
    score = mean_squared_error(ds[:,-1], predictions)
    print(score)

    ax = range(len(predictions))
    # plt.scatter(ax, ds[:,-1], s=1, c = 'blue')
    # plt.scatter(ax, predictions, s = 0.5, c='red')
    # plt.scatter(ax, llist, s = 0.5, c='red')
    # plt.scatter(ax,prequential,s = 0.5, c='black')
    # plt.xlabel('Sample Stream',fontweight='bold')
    # plt.ylabel('y unit',fontweight='bold')
    # plt.show()


    fig1 = plt.figure(figsize=(6, 4))  # Figure for blue and red points
    fig2 = plt.figure(figsize=(6, 4))  # Figure for black points
    fig3 = plt.figure(figsize=(6, 4))  # Figure for black points
    fig4 = plt.figure(figsize=(6, 4))
    fig5 = plt.figure(figsize=(6, 4))
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()
    ax3 = fig3.add_subplot()
    ax4 = fig4.add_subplot()
    ax5 = fig5.add_subplot()
    ax1.scatter(ax, ds[:,-1], s=1, c='blue', label='Actual')
    ax1.scatter(ax, predictions, s=0.5, c='red', label='Predicted')
    ax1.set_xlabel('Sample Stream', fontweight='bold')
    ax1.set_ylabel('y unit', fontweight='bold')
    ax1.legend()
    ax2.scatter(ax, prequential, s=0.5, c='black', label='Prequential error')
    ax2.scatter(ax, onlineWMAPE, s=0.5, c='red', label='Online WMAPE')
    ax2.set_xlabel('Sample Stream', fontweight='bold')
    ax2.set_ylabel('y unit', fontweight='bold')
    ax2.legend()
    ax3.scatter(ax, ds[:,-1] - predictions, s=0.5, c='black', label='Residuals')
    ax3.plot([0.0,len(ds[:,-1] - predictions)],[0.0,0.0],'k')
    ax3.set_xlabel('Sample Stream', fontweight='bold')
    ax3.set_ylabel(r'y - $\hat{y}$', fontweight='bold')
    ax3.legend()
    ax4.scatter(predictions, ds[:,-1], s=1, c = 'blue')
    ax4.plot([0.0,math.ceil(max(ds[:,-1]))],[0.0,math.ceil(max(ds[:,-1]))],'k')
    ax4.set(xlabel='Predicted')
    ax4.set(ylabel='Actual')
    ax4.legend()
    ax5.scatter(ax, llist, s=1, c = 'blue')
    ax5.set(xlabel='Stream')
    ax5.set(ylabel='Ensemble size')
    ax5.legend()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
