
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
    def __init__(self, n_estimators=10, beta=0.5, theta=0.1, min_estimators = 10, learning_rate=1, scaler = 1.0):
        self.n_estimators = n_estimators
        self.min_estimators = min_estimators
        self.beta = beta
        self.theta = theta
        self.buffer_X = []
        self.buffer_y = []
        self.window_size = 50
        self.learning_rate = learning_rate  # \lambda do boosting Oza
        self.scaler = scaler # \lambda para soma do ensemble Hastie
        self.regressors = []
        self.weights = []
        for _ in range(n_estimators):
            regressor = tree.HoeffdingTreeRegressor(leaf_prediction='mean',max_depth=3) # depht = 1 stump #Hoeffding Adaptive Tree regressor (HATR)
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


    def partial_fit(self, X, y, L=0):
        residuo = y
        pred = 0
        mu = 1
        for i, regressor in enumerate(self.regressors):  # ver alternativa para não atualizar todos
            k = poisson.rvs(mu)
            for _ in range(k):
                regressor.learn_one(dict(enumerate(X)), residuo)
            pred += self.scaler * regressor.predict_one(dict(enumerate(X)))
            residuo -= pred
            error = abs(residuo - pred)
            #self.weights[i] = (1 - self.beta) * self.weights[i] + self.beta * error #(y / error)
            mu = int(self.learning_rate * error)
            # teste
            if residuo < 0:
                break

        # if self.n_estimators > self.min_estimators:
        #     try:
        #         filtered = [(r, w) for r, w in zip(self.regressors, self.weights) if w > self.theta]
        #         self.regressors, self.weights = map(list, zip(*filtered))
        #         self.n_estimators = len(self.weights)
        #     except ValueError:
        #         print(f'Nenhum regressor tem peso maior que {self.theta}.')

        if L > 0:
            for _ in range(L):
                regressor = tree.HoeffdingTreeRegressor(leaf_prediction='mean', max_depth=3)
                self.regressors.append(regressor)
                self.weights.append(1.0)
            self.n_estimators = len(self.regressors) #len(self.weights)



    def partial_fit_Ens_old(self, X, y, p, L=0):  #p - ensemble predicton
        residuo = y - p   # TESTES real - pred
        mu = 1
        for i, regressor in enumerate(self.regressors):  # ver alternativa para não atualizar todos - mais recentes
            k = poisson.rvs(mu)
            #print(k)
            for _ in range(k):
                regressor.learn_one(dict(enumerate(X)), residuo)
            pred = self.scaler * regressor.predict_one(dict(enumerate(X)))
            residuo = residuo - pred


    def get_n_estimators(self):
        return self.n_estimators

    def partial_fit_Ens(self, X, y, p, L=0, erroLimiar= 2.0):  # p - ensemble prediction
        residuo = y - p  # TESTES real - pred
        mu = 1
        percentUpdate = 10

        if L == 0:
            percentUpdate = 1
            residuo = y

        total_regressors = len(self.regressors)
        num_recent_regressors = max(1, total_regressors // percentUpdate)  # Calcula os últimos 10%

        # Itera sobre os últimos 10% dos regressores
        for i, regressor in enumerate(self.regressors[-num_recent_regressors:]):
            k = poisson.rvs(mu)
            #print(k)
            for _ in range(k):
                regressor.learn_one(dict(enumerate(X)), residuo)
            pred = self.scaler * regressor.predict_one(dict(enumerate(X)))
            residuo = residuo - pred
            error = abs(residuo - pred)  # wmape
            #mu = int(self.learning_rate * error)

        # Atualizar o buffer com a nova instância
        self.buffer_X.append(X)
        self.buffer_y.append(y)

        # Manter apenas os últimos `window_size` elementos no buffer
        if len(self.buffer_X) > self.window_size:
            self.buffer_X.pop(0)
            self.buffer_y.pop(0)

        # Remover os regressores com erro maior que erroLimiar
        #if self.n_estimators > 200: #len(self.buffer_X) == self.window_size:
            #self.remove_least_contributing_regressors(self.buffer_X, self.buffer_y) #           self.remove_worst_regressors(erroLimiar)
        #    self.remove_least_contributing_regressors_percent(self.buffer_X, self.buffer_y)

        if L > 0:
            for _ in range(L):
                regressor = tree.HoeffdingTreeRegressor(leaf_prediction='mean', max_depth=2)
                self.regressors.append(regressor)
                self.weights.append(1.0)
            self.n_estimators = len(self.regressors) #len(self.weights)


    def remove_worst_regressors(self, erroLimiar):
        # Inicializar lista de erros para cada regressor
        wmape_errors = [0] * len(self.regressors)
        X_window = self.buffer_X
        y_window = self.buffer_y

        # Calcular predições e WMAPE para cada regressor na janela
        for j, regressor in enumerate(self.regressors):
            preds = [self.scaler * regressor.predict_one(dict(enumerate(x))) for x in X_window]
            absolute_errors = [abs(y_true - y_pred) for y_true, y_pred in zip(y_window, preds)]
            wmape = sum(absolute_errors) / sum(y_window)
            wmape_errors[j] += wmape

        # Combinar regressores e erros WMAPE
        regressor_errors = list(zip(self.regressors, wmape_errors))

        # Filtrar os regressores com erro maior que erroLimiar
        self.regressors = [regr for regr, err in regressor_errors if err <= erroLimiar]
        self.weights = [1.0] * len(self.regressors)  # Atualizar pesos para novos regressores
        print(max(wmape_errors))
        print(len(wmape_errors) - len(self.regressors))


    def remove_least_contributing_regressors(self, X, y): # X e y são de uma janela
        total_error = np.sum(np.abs(y - self.predict_ensemble(X)))

        # Calcular a contribuição de cada regressor
        contributions = []
        for i, regressor in enumerate(self.regressors):
            # Predições sem o regressor atual
            predictions_without_regressor = [
                self.scaler * regressor.predict_one(dict(enumerate(x))) for idx, x in enumerate(X) if idx != i
            ]
            error_without_regressor = np.sum(np.abs(y - np.sum(predictions_without_regressor, axis=0)))
            contribution = total_error - error_without_regressor
            contributions.append((i, contribution))

        # Ordenar os regressores pela contribuição (ascendente)
        contributions.sort(key=lambda x: x[1])

        # Número de regressores a remover (10% do total)
        num_to_remove = max(1, len(self.regressors) // 10)

        # Remover os regressores com menor contribuição
        indices_to_remove = [idx for idx, _ in contributions[:num_to_remove]]
        self.regressors = [regr for idx, regr in enumerate(self.regressors) if idx not in indices_to_remove]

        print(f'Removidos {num_to_remove} regressores com menor contribuição.')



    def predict_ensemble(self, X):
    # Prever com todos o ensemble
        return np.sum([self.scaler * regressor.predict_one(dict(enumerate(x))) for regressor in self.regressors for x in X], axis=0)


    def remove_least_contributing_regressors_percent(self, X, y, threshold=0.05):
        total_error = np.sum(np.abs(y - self.predict_ensemble(X)))

        # Calcular a contribuição de cada regressor
        contributions = []
        for i, regressor in enumerate(self.regressors):
            # Predições sem o regressor atual
            predictions_without_regressor = [
                self.scaler * regressor.predict_one(dict(enumerate(x))) for idx, x in enumerate(X) if idx != i
            ]
            error_without_regressor = np.sum(np.abs(y - np.sum(predictions_without_regressor, axis=0)))
            contribution = (total_error - error_without_regressor) / total_error
            contributions.append((i, contribution))

        # Filtrar os regressores com contribuição inferior ao limiar
        indices_to_remove = [idx for idx, contrib in contributions if contrib < threshold]
        self.regressors = [regr for idx, regr in enumerate(self.regressors) if idx not in indices_to_remove]

        print(f'Removidos {len(indices_to_remove)} regressores com contribuição inferior a {threshold*100}%.')




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

    def predictTest(self, X):
        """
        Predicts target values for new data.
        :param X: Feature matrix of shape (n_samples, n_features).
        :return: Predicted target values.
        """
        y_pred = 0
        for i, learner in enumerate(self.regressors):
            y_pred += self.scaler * learner.predict_one(dict(enumerate(X)))
        return y_pred

    def predict1(self, X):
        if not self.regressors:
            raise Exception("Regressor não ajustado")
        if X.ndim == 1:     ## Ver mean, tentar alternativas como media ponderada com os pesos dos regressores
            predictions = np.sum([self.scaler*regressor.predict_one(dict(enumerate(X))) for regressor in self.regressors])
        else:   # era mean em sum
            predictions = np.sum([self.scaler*np.mean([regressor.predict_one(dict(enumerate(xi))) for regressor in self.regressors]) for xi in X])
        return predictions


    def predict(self, X, y):
        predictions = [self.scaler*regressor.predict_one(dict(enumerate(X))) for regressor in self.regressors]
        boostPred = np.sum(predictions)
        for i, (regressor, pred) in enumerate(zip(self.regressors, predictions)):
            error = abs(y - pred)/y  # Calcula o erro absoluto
            self.weights[i] = (1 - self.beta) * self.weights[i] + self.beta * abs(boostPred-y)/error  # Atualiza o peso para evitar divisão por zero

        if self.n_estimators > self.min_estimators:
            try:
                filtered = [(r, w) for r, w in zip(self.regressors, self.weights) if w > self.theta]
                self.regressors, self.weights = map(list, zip(*filtered))
                self.n_estimators = len(self.weights)
            except ValueError:
                print(f'Nenhum regressor tem peso maior que {self.theta}.')

        return boostPred

    def predictAndUpdate(self, X,y):
        predictions = [self.scaler*regressor.predict_one(dict(enumerate(X))) for regressor in self.regressors]
        boostPred = np.sum(predictions)
        EnsAbsSum = np.sum(np.abs(predictions))  # soma dos valores absolutos
        for i, regressor in enumerate(self.regressors):
            #if predictions[i]*y > 0 :
                self.weights[i] = (1 - self.beta) * self.weights[i] + self.beta * np.abs(predictions[i])/EnsAbsSum  # Atualiza o peso para evitar divisão por zero
            #else:
            #    self.weights[i] = (1 - self.beta) * self.weights[i] + self.beta * np.abs(1/(1+predictions[i]))/EnsAbsSum

        if self.n_estimators > self.min_estimators:
            try:
                filtered = [(r, w) for r, w in zip(self.regressors, self.weights) if w > self.theta]
                self.regressors, self.weights = map(list, zip(*filtered))
                self.n_estimators = len(self.weights)
            except ValueError:
                print(f'Nenhum regressor tem peso maior que {self.theta}.')

        return boostPred


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