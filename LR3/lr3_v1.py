import numpy as np
from numpy.linalg import inv
from termcolor import colored


def check_nash_opt(bimatrix, i, j):
    matrix_dim, _ = bimatrix[0].shape
    best_a_strategy = False
    best_b_strategy = False

    for iterI in range(matrix_dim):
        if bimatrix[0][iterI][j] > bimatrix[0][i][j]:
            best_b_strategy = True
        if bimatrix[1][i][iterI] > bimatrix[1][i][j]:
            best_a_strategy = True

    return not (best_a_strategy or best_b_strategy)


def check_pareto_eff(bimatrix, i, j):
    matrix_dim, _ = bimatrix[0].shape
    best_strategy = False

    for iIter in range(matrix_dim):
        for jIter in range(matrix_dim):
            if (
                (bimatrix[0][iIter][jIter] > bimatrix[0][i][j] and
                 bimatrix[1][iIter][jIter] >= bimatrix[1][i][j]) or
                (bimatrix[1][iIter][jIter] > bimatrix[1][i][j] and
                 bimatrix[0][iIter][jIter] >= bimatrix[0][i][j])
            ):
                best_strategy = True

    return not best_strategy


def print_analitic_result(A_matrix, B_matrix):
    print("Analitic method:")
    u = np.array([[1,1]])
    ut = u.transpose()
    Bt = np.linalg.inv(B_matrix)
    At = np.linalg.inv(A_matrix)
    A_iter = np.dot(u,At)
    B_iter = np.dot(u,Bt)
    v1 = np.divide(1,np.dot(A_iter,ut))
    v2 = np.divide(1,np.dot(B_iter,ut))
    x = np.dot(np.dot(v2[0][0],u),Bt)
    y = np.dot(np.dot(v1[0][0],At),ut)
    print("x = ",x)
    print("y = ",y)
    print("v1 = ",round(v1[0][0],3))
    print("v2 = ",round(v2[0][0],3))



def print_result(bimatrix):
    matrix_dim, _ = bimatrix[0].shape

    for i in range(matrix_dim):
        line = []
        for j in range(matrix_dim):
            is_nash_opt = check_nash_opt(bimatrix, i, j)
            is_pareto_eff = check_pareto_eff(bimatrix, i, j)

            point_str = "({0: <3}|{1: <3})".format(bimatrix[0][i][j], bimatrix[1][i][j])
            if is_pareto_eff and is_nash_opt:
                line.append(colored(point_str, 'red'))
            elif is_pareto_eff:
                line.append(colored(point_str, 'green'))
            elif is_nash_opt:
                line.append(colored(point_str, 'blue'))
            else:
                line.append(point_str)

        print(" ".join(line))
    print()


if __name__ == "__main__":
    crosswayBiMatrix = np.array([
        [[1, 0.5],
         [2, 0]],
        [[1, 2],
         [0.5, 0]],
    ], float)

    familyDisputeBiMatrix = np.array([
        [[4, 0],
         [0, 1]],
        [[1, 0],
         [0, 4]],
    ], float)

    prisonerDilemmaBiMatrix = np.array([
        [[-5, 0],
         [-10, -1]],
        [[-5, -10],
         [0, -1]],
    ], float)

    MAX_VALUE = 50
    MIN_VALUE = 0
    DIM = 10
    randomBiMatrix = np.array([
        np.random.randint(MIN_VALUE, MAX_VALUE, (DIM, DIM)),
        np.random.randint(MIN_VALUE, MAX_VALUE, (DIM, DIM)),
    ], int)

    BiMatrixVariant = np.array([
        [[4, 6],
         [11, 0]],
        [[1, 2],
         [7, 5]],
    ], float)
    A_matrix_variant = np.array([[4,6],[11,0]])
    B_matrix_variant = np.array([[1,2],[7,5]])


    print(colored('Парето эффективность и равновесие по Нэшу.', 'red'))
    print(colored('Парето эффективность.', 'green'))
    print(colored('Равновесие по Нэшу.', 'blue'))
    print()

    print("____________________ПРОВЕРКА АЛГОРИТМА____________________")
    print("Перекресток со смещением:")
    print_result(crosswayBiMatrix)

    print("Семейный спор:")
    print_result(familyDisputeBiMatrix)

    print("Дилемма заключенного:")
    print_result(prisonerDilemmaBiMatrix)

    print("____________________РЕШЕНИЯ НА МАТРИЦАХ____________________")

    print("Случайная матрица 10х10:")
    print_result(randomBiMatrix)

    print("Матрица варианта №13:")
    print_result(BiMatrixVariant)
    print_analitic_result(A_matrix_variant, B_matrix_variant)
