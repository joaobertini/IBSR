# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IBSR import IBSRegressor

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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DATASET_PATH = 'housing.csv'

    ds = streamData(DATASET_PATH)

    ibs =IBSRegressor()  # cria um modelo base
    ibs.fit(ds[0,:-1],ds[0,-1])  # treina com o primeiro exemplo
    ds = np.delete(ds, 0, axis=0)  # remove o exemplo usado

    performance_scores = [0]
    denominador = 1
    predictions = []
    prequential = []
    alpha = 0.999  # prequential forgetting parameter
    cont = 1 # conta exemplos processados para calculo do MAE
    erro = 0
    ys = 0

    llist = []

    # simulando um fluxo de dados, testa e treina
    for row in ds:

        erroNominador = 0
        ##### testa ###
        predictions.append(ibs.predict(row[:-1]))  # lista das predições

        # prequential
        erro += abs(ibs.predict(row[:-1]) - row[-1])  # erro da instancia atual - MAE
        ys += abs(row[-1]) # soma iterativa dos ys


        ## P(20000) = al20000 ...  al^3 x e1/al^3 + al^2 x e2/al^2 + al^1 x e3/al^1 + al^0 x e4/al^0

        #errmae = erro/cont
        #mae = mean_absolute_error(actual, pred)
        #wmape = mae * len(pred)*100/sum(abs(actual))
        errmae = erro/ys  # online wmape

        array = np.array(performance_scores) * alpha
        erroNominador = errmae + np.sum(array)
        prequential.append(erroNominador/denominador)
        denominador += alpha

        # erro wmape online
        performance_scores = list(array)
        performance_scores.append(errmae) # erro
        cont += 1

        #calculo para adição de regressores
        inner_term = math.exp(prequential[-1])
        L = math.ceil(inner_term)

        # print(ibs.get_n_estimators()) # imprime tamanho do ensemble para teste
        llist.append(ibs.get_n_estimators())

        ### treina ####
        # atualiza online - Poisson
        ibs.partial_fit(row[:-1], row[-1],L)

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
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()
    ax1.scatter(ax, ds[:,-1], s=1, c='blue', label='Actual')
    ax1.scatter(ax, predictions, s=0.5, c='red', label='Predicted')
    ax1.set_xlabel('Sample Stream', fontweight='bold')
    ax1.set_ylabel('y unit', fontweight='bold')
    ax1.legend()
    ax2.scatter(ax, prequential, s=0.5, c='black', label='Prequential error')
    ax2.set_xlabel('Sample Stream', fontweight='bold')
    ax2.set_ylabel('y unit', fontweight='bold')
    ax2.legend()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
