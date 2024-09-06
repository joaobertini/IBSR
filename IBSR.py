
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from river import tree
from scipy.stats import poisson
## https://riverml.xyz/dev/api/tree/HoeffdingTreeRegressor/


#______________
class IBSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=20, beta=0.5, theta=0.5, min_estimators = 10, learning_rate=0.1, scaler = 0.1):
        self.n_estimators = n_estimators
        self.min_estimators = min_estimators
        self.beta = beta
        self.theta = theta
        self.learning_rate = learning_rate  # \lambda do boosting Oza
        self.scaler = scaler # \lambda para soma do ensemble Hastie
        self.regressors = []
        self.weights = []
        for _ in range(n_estimators):
            regressor = tree.HoeffdingTreeRegressor(leaf_prediction='mean',max_depth=2) # depht = 1 stump #Hoeffding Adaptive Tree regressor (HATR)
            self.regressors.append(regressor) # adiciona regressor ao ensemble
            self.weights.append(1.0)

    def fit(self, X, y):

        if X.ndim == 1:   # veridica se um unico elemento foi passado
            for reg in self.regressors:
                reg.learn_one(dict(enumerate(X)),y)
        else:            # se mais de um elemento, treina um a um
            for reg in self.regressors:
                for xi, yi in zip(X, y):
                    reg.learn_one(dict(enumerate(xi)), yi)


    def partial_fit(self, X, y, L = 0):  # para um exemplo por vez
        mu = 1
        residuo = y
        pred = 0
        #for reg, wei in zip(self.regressors, self.weights): # reg in self.regressors:  # online boosting
        for i in range(len(self.regressors)):
            k = poisson.rvs(mu)
            for _ in range(k):
                self.regressors[i].learn_one(dict(enumerate(X)),residuo)

            pred = pred + self.scaler*self.regressors[i].predict_one(dict(enumerate(X)))  # Hastie Eq. 8.10
            residuo = residuo - pred                # eq. 8.11
            error = abs(residuo - pred) # erro para determinar mu do proximo regressor
            self.weights[i] = (1-self.beta)*self.weights[i] + self.beta*(y/error)  # y/pred
            ## atualizar wei nos pesos

            mu = (int)(self.learning_rate*error)  # atualiza mu, ver valores
            #print(mu)

        # remove regressores -- ver se atualização de wei deve ser erro ou scaler*pred...
        if self.n_estimators > self.min_estimators:                             #    elimina w < theta
            try:  # verifiar se pode remover todos - deixar um aleatorio -- VER
                filtered = [(r, w) for r, w in zip(self.regressors, self.weights) if w > self.theta]
                self.regressors, self.weights = map(list, zip(*filtered))
                # estava retornando em tupla
                #self.regressors, self.weights = zip(*[(r, w) for r, w in zip(self.regressors, self.weights) if w > self.theta])

                n_estimators = len(self.weights)
            except ValueError as ve:
                print(f'Nenhum regressor tem peso maior que {self.theta}.')

            #errors = np.clip(errors, 1e-10, 1e10) # limitar valores nos erros
        if L > 0:  # L é o número de regressores a adicionar
            for _ in range(L):
                regressor = tree.HoeffdingTreeRegressor(leaf_prediction='mean',max_depth=2) # depht = 1 stump #Hoeffding Adaptive Tree regressor (HATR)
                self.regressors.append(regressor) # adiciona regressor ao ensemble
                self.weights.append(1.0)
            self.n_estimators = len(self.weights)


    def get_n_estimators(self):
        return self.n_estimators


        # # Calcular e limitar o erro ponderado
        # weighted_errors = [np.clip(np.mean(errors * w), 1e-10, 1e10) for w in self.weights]
        # self.weights = [(1 - self.beta) * w + self.beta * e for w, e in zip(self.weights, weighted_errors)]
        #
        # # Remover regressores com peso abaixo do limiar
        # self.regressors, self.weights = zip(*[(r, w) for r, w in zip(self.regressors, self.weights) if w > self.theta])
        # self.regressors = list(self.regressors)
        # self.weights = list(self.weights)
        #
        #
        # # atualiza regressores existentes
        # for regressor in self.regressors:
        #     #regressor = tree.HoeffdingTreeRegressor()
        #     if X.ndim == 1:   # veridica se um unico elemento foi passado
        #         regressor.learn_one(dict(enumerate(X)),y)
        #     else:    # treina um a um
        #         for xi, yi in zip(X, y):
        #             regressor.learn_one(dict(enumerate(xi)), yi)
        #         #self.regressors.append(regressor)
        #         #self.weights.append(1.0)

        # Adicionar novos regressores - batch ou online

    def predict(self, X):
        if not self.regressors:
            raise Exception("Regressor não ajustado")
        if X.ndim == 1:     ## Ver mean, tentar alternativas como media ponderada com os pesos dos regressores
            predictions = np.sum([self.scaler*regressor.predict_one(dict(enumerate(X))) for regressor in self.regressors])
        else:
            predictions = np.sum([self.scaler*np.mean([regressor.predict_one(dict(enumerate(xi))) for regressor in self.regressors]) for xi in X])
        return predictions

    # def prequential_evaluation(self, X, y, metric=mean_absolute_error):
    #     if len(X) != len(y):
    #         raise ValueError("X e y devem ter o mesmo tamanho!")
    #
    #     performance_scores = []
    #
    #     for i in range(len(X)):
    #         X_test, y_test = X[i:i+1], y[i:i+1]
    #
    #         if i > 0:
    #             prediction = self.predict(X_test)
    #             score = metric(y_test, prediction)
    #             performance_scores.append(score)
    #
    #         self.partial_fit(X_test, y_test)
    #
    #     return performance_scores


#     def fuzzy_adjust(self, error, n_regressors):
#         error_high = fuzzy_high(error, 0.1)
#         error_low = fuzzy_low(error, 0.05)
#         error_medium = fuzzy_medium(error, 0.05, 0.1)
#
#         regressors_high = fuzzy_high(n_regressors, 10)
#         regressors_low = fuzzy_low(n_regressors, 5)
#         regressors_medium = fuzzy_medium(n_regressors, 5, 10)
#
#         if error_high and regressors_high:
#             self.n_estimators = max(1, self.n_estimators - 1)
#             self.beta = min(1, self.beta + 0.1)
#             self.theta = min(1, self.theta + 0.1)
#         elif error_high and regressors_low:
#             self.n_estimators = self.n_estimators + 1
#             self.theta = max(0, self.theta - 0.1)
#         elif error_low and regressors_medium:
#             self.n_estimators = self.n_estimators
#         elif error_low and regressors_high:
#             self.n_estimators = max(1, self.n_estimators - 1)
#             self.beta = min(1, self.beta + 0.1)
#             self.theta = min(1, self.theta + 0.1)
#
# # Funções de pertinência fuzzy (regras)
#     def fuzzy_low(value, threshold):
#         return max(0, (threshold - value) / threshold)
#
#     def fuzzy_high(value, threshold):
#         return max(0, (value - threshold) / threshold)
#
#     def fuzzy_medium(value, low_threshold, high_threshold):
#         if value < low_threshold:
#             return 0
#         elif value > high_threshold:
#             return 0
#         else:
#             return (value - low_threshold) / (high_threshold - low_threshold)