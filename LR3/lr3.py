import numpy as np
import pandas as pd
import os
import random

def calc_nesh(matrix_input):
    result = []
    for i in range(0,len(matrix_input)):
        max_1 = -51
        max_2 = -51
        max_1_j = 0
        max_2_j = 0
        for j in range(0,len(matrix_input[i])):
            val = matrix_input[j][i]
            if val[0] > max_1:
                max_1 = val[0]
                max_1_j = j

        for i in range(0,len(matrix_input[max_1_j])):
            val = matrix_input[max_1_j][i]
            if val[1] > max_2:
                max_2 = val[1]
                max_2_j = i
    
        if max_1 == matrix_input[max_1_j][max_2_j][0]:
            result.append([max_1,max_2])
    
    if len(result) != 0:
        return result
    else:
        print("Nesh not found\n")

def calc_pareto(matrix_input):
    cheak_list = []
    cheak_list_max = []
    cheak_list_max_iter = []
    result = []
    new_result = []
    for i in range(0,len(matrix_input)):
        for j in range(0,len(matrix_input[i])):
            val = matrix_input[i][j]
            for k in range(0,len(matrix_input)):
                for m in range(0,len(matrix_input[k])):
                    val_iter = matrix_input[k][m]
                    if (val_iter[0] >= val[0]) and (val_iter[1] >= val[1]):
                        val = val_iter

            if (val[0] == matrix_input[i][j][0]) and (val[1] == matrix_input[i][j][1]):
                cheak_list.append(val)
    
    max_1 = -51
    max_2 = -51

    for i in range(0,len(matrix_input)):
        for j in range(0,len(matrix_input[i])):
            val = matrix_input[i][j]
            if val[0] > max_1:
                max_1 = val[0]
            if val[1] > max_2:
                max_2 = val[1]

    for i in range(0,len(matrix_input)):
        for j in range(0,len(matrix_input[i])):
            val = matrix_input[i][j]
            if (max_1 == val[0]) or (max_2 == val[1]):
                cheak_list_max_iter.append(val)
    
    for vl in cheak_list_max_iter:
        for vl1 in cheak_list_max_iter:
            if vl[0] >= vl1[0] and vl[1] >= vl1[1]:
                cheak_list_max.append(vl)

    for val in cheak_list_max:
        result.append(val)
    for val1 in cheak_list:
        result.append(val) 

    return result


if __name__ == "__main__":
    N = 10
    matrix = []
    steps_broun_rob = 3000
    for j in range(0,N):
        iter_arr = []
        for i in range(0,N):
            iter_val = []
            iter_val.append(random.randint(-50, 50))
            iter_val.append(random.randint(-50, 50))
            iter_arr.append(iter_val)
        matrix.append(iter_arr)

    matrix_challenge_robber = [[[-5,-5],[0,-10]],[[-10,0],[-1,-1]]]
    print("Challenge_robber: ",matrix_challenge_robber)
    print("Nesh: ",calc_nesh(matrix_challenge_robber))
    print("Pareto: ",calc_pareto(matrix_challenge_robber))

    matrix_family_challenge = [[[4,1],[0,0]],[[0,0],[1,4]]]
    print("Challenge_family: ",matrix_family_challenge)
    print("Nesh: ",calc_nesh(matrix_family_challenge))
    print("Pareto: ",calc_pareto(matrix_family_challenge))

    matrix_family_crossroads = [[[1,1],[1,2]],[[2,1],[0,0]]]
    print("Challenge_crossroads: ",matrix_family_crossroads)
    print("Nesh: ",calc_nesh(matrix_family_crossroads))
    print("Pareto: ",calc_pareto(matrix_family_crossroads))

    print(matrix)
    print("Nesh: ",calc_nesh(matrix))
    print("Pareto: ",calc_pareto(matrix))

    matrix_variant = [[[4,1],[6,2]],[[11,7],[0,5]]]

    print("Nesh: ",calc_nesh(matrix_variant))

    A = np.array([[4,6],[11,0]])
    B = np.array([[1,2],[7,5]])

    u = np.array([[1,1]])
    ut = u.transpose()
    Bt = np.linalg.inv(B)
    At = np.linalg.inv(A)
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



