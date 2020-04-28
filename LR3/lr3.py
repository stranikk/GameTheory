import numpy as np
from numpy.linalg import inv, det
import pandas as pd
import random

class Game:
    __game = []
    __nesh = []
    __pareto = []

    def add_game(self, input_game):
        self.__game = input_game
    
    def get_game(self):
        return self.__game

    def get_nesh(self):
        return self.__nesh
    
    def get_pareto(self):
        return self.__pareto

    def optimal_strat(self, row, col):
        item = self.__game[row, col]
        for i in range(self.__game.shape[0]):
            for j in range(self.__game.shape[1]):
                current = self.__game[i, j]
                if (item < current).any() and (item <= current).all():
                    return False
        return True

    def pareto_optimal(self):
        optimal_strategies = []
        for i in range(self.__game.shape[0]):
            for j in range(self.__game.shape[1]):
                if self.optimal_strat(i, j):
                    optimal_strategies.append((i, j))
                
        self.__pareto = optimal_strategies

    def count_maximum(self,lst):
        indexes = np.linspace(0, lst.shape[0] - 1, lst.shape[0], dtype = int)   
        return indexes[np.isin(lst, np.max(lst))]

    def maximum_1(self, i):
        indexes = self.count_maximum(self.__game[:, i, 0])
        return indexes[self.__game[indexes, i, 0] >= np.max(self.__game[indexes, i, 0])]

    def maximum_2(self, i):
        indexes = self.count_maximum(self.__game[i, :, 1])
        return indexes[self.__game[indexes, i, 1] >= np.max(self.__game[indexes, i, 1])]

    def nash_optimal(self):
        optimal_strategies = []
        for i in range(self.__game.shape[0]):
            second_ids = self.maximum_2(i)
            for j in second_ids:
                first_ids = self.maximum_1(j)
                if i in first_ids:
                    optimal_strategies.append((i, j))
        self.__nesh = optimal_strategies

    def format_game(self):
        formatted_array = [["(%s | %s)" % (str(j[0]), str(j[1])) for j in i] for i in self.__game]
        df = pd.DataFrame(formatted_array)
        return df

def print_game(game):
    main_game = Game()
    main_game.add_game(game)
    main_game.nash_optimal()
    main_game.pareto_optimal()
    game = main_game.get_game()

    nesh = []
    pareto = []
    for val in main_game.get_nesh():
        x = val[0]
        y = val[1]
        nesh.append(game[x][y])

    for val in main_game.get_pareto():
        x = val[0]
        y = val[1]
        pareto.append(game[x][y])

    print("Game: ", main_game.format_game())
    print("Nash: ", nesh)
    print("Pareto: ", pareto)

if __name__ == "__main__":
    print("____________________ПРОВЕРКА АЛГОРИТМА____________________")
    
    print("Перекресток со смещением:")
    epsilon = round(random.random(), 3)
    z = round(random.random(), 3)
    crosswayBiMatrix = np.array(
        [[[1, 1], [1 - epsilon, 2]],
        [[2, 1 - z], [0, 0]]])
    print_game(crosswayBiMatrix)

    print("Семейный спор:")
    familyDisputeBiMatrix = np.array(
        [[[4, 1], [0, 0]],
        [[0, 0], [1, 4]]]
    )
    print_game(familyDisputeBiMatrix)

    print("Дилемма заключенного:")
    prisonerDilemmaBiMatrix = np.array(
        [[[ -5, -5], [ 0, -10]],
        [[-10,  0], [-1,  -1]]]
    )
    print_game(prisonerDilemmaBiMatrix)

    print("____________________РЕШЕНИЯ НА МАТРИЦАХ____________________")
    print("Случайная матрица 10х10:")
    game = np.random.randint(-50, 50, (10, 10, 2))
    print_game(game)
    
    print("Матрица варианта №13:")
    BiMatrixVariant = np.array([
        [[ 4, 1], [6, 2]],
        [[11, 7], [0, 5]]
    ])
    print_game(BiMatrixVariant)

    A = BiMatrixVariant[:,:,0]
    B = BiMatrixVariant[:,:,1]
    u = np.ones(2)

    v1 = 1/(u.dot(inv(A)).dot(u))
    v2 = 1/(u.dot(inv(B)).dot(u))

    x = v2 * u.dot(inv(B))
    y = v1 * inv(A).dot(u)

    print('A:')
    print(pd.DataFrame(A))
    print('B:')
    print(pd.DataFrame(B))
    print()
    print('v1:', round(v1, 3))
    print('v2:', round(v2, 3))

    print('x:', np.round(x, 3))
    print('y:', np.round(y, 3))
