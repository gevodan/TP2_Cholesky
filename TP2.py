#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      arthu
#
# Created:     31/03/2022
# Copyright:   (c) arthu 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import math
import matplotlib.pyplot as plt
import copy as c

def Cholesky(A):    #Retourne la matrice de Cholesky à condition que la matrice de départ soit une matrice symétrique défiie positive
    n=len(A)
    Q = 0   # calcule le nombre total d'opérations sur les matrices
    L = np.zeros((n, n))
    for j in range (n) :
        s=0
        for k in range (j) :
            s = s + L[j, k]**2
            Q = Q + 1
        L[j, j] = math.sqrt(A[j, j] - s)
        Q = Q + 1
        for i in range (j+1, n) :
            s = 0
            for k in range (j) :
                s = s + L[i, k]*L[j, k]
                Q = Q + 1
            L[i, j] = (A[i, j] - s) / L[j, j]
            Q = Q + 1
    return L, Q




def ResolCholesky (A, B, Q=0) :
    n = len(A)
    L = Cholesky(A)[0]
    Q = Cholesky(A)[1]  # calcule le nombre d'opérations sur les matrices
    T = L.transpose()   # transposée de la matrice L
    Y = np.zeros(n)     # vecteur colonne Y
    X = np.zeros(n)     # vecteur colonne X
    Y[0] = B[0] / L[0, 0]
    k = 1   # permet de rester sur la même ligne dans la boucle j dans la double boucle sur Y
    l = n-2   # permet de rester sur la même ligne dans la boucle j dans la double boucle sur X
    for i in range (1, n) :     # cette double boucle permet de calculer Y
        s = 0
        for j in range (1, n) :
            s = s + L[k, j-1]*Y[j-1]
            Q = Q + 1
        Y[i] = (B[i] - s) / L[i, i]
        Q = Q + 1
        k = k + 1
    X[n-1] = Y[n-1] / T[n-1, n-1]
    for i in range (n-2, -1, -1) :
        s = 0
        for j in range (n-2, -1, -1) :
            s = s + T[l, j+1]*X[j+1]
            Q = Q + 1
        X[i] = (Y[i] - s) / T[i, i]
        Q = Q + 1
        l = l - 1
    return Y, X, Q


def ReductionGauss(M):
    Q = 0
    n = M.shape[0]

    for j in range(0,n - 1):
        if M[j,j] == 0:
            return "Erreur"
        for i in range (j + 1 , n):
            # L[i,j] = M[i,j]/M[j,j]
            M[i,] = M[i,] - M[i,j]/M[j,j]*M[j,]
            Q = Q + 1
    return M, Q



def ResolutionSystTrigSup(Taug, Q=0):

    K = c.deepcopy(Taug)
    n = len(K)
    x = np.zeros((n,1))
    x[n-1] = K[n-1][n]/K[n-1][n-1]
    Q = Q + 1

    for i in range (n-2, -1, -1):
        x[i] = K[i][n]
        Q = Q + 1

        for j in range (i+1, n):
            x[i] = x[i] - K[i][j] * x[j]
            Q = Q + 1

        x[i] = x[i] / K[i][i]
        Q = Q + 1


    X = np.asarray(x).reshape(n, 1)

    return X, Q

def Gauss(A,B, n):
    n = len(A)
    b = B.reshape(n,1)

    Taug = np.concatenate((A,b), axis = 1)
    Taug = ReductionGauss(Taug)[0]
    Q = ReductionGauss(Taug)[1]

    return ResolutionSystTrigSup(Taug, Q)


epsilon = 1e-16
y1 = np.zeros(10)   # matrice ligne représentant le nombre de calculs matriciels en fonction du nombre de lignes
y2 = np.zeros(10)   # matrice représentant les erreurs en fonction du nombre de lignes
tableau1 = []
tableau2 = []
for i in range (1, 11) :
    y3 = 0
    y4 = 0
    Epsilon = np.zeros(i)
    B = np.random.rand(i)
    for j in range (0, i) :
        Epsilon[j] = epsilon    # on crée une matrice ligne avec j éléments uniquement composée de 10^-16
    A = 2*np.eye(i) - np.diag(np.ones(i-1),1) - np.diag(np.ones(i-1), -1)
    C = ResolCholesky(A, B)
    A = 2*np.eye(i) - np.diag(np.ones(i-1),1) - np.diag(np.ones(i-1), -1)
    D = Gauss(A, B, i)
    y1[i-1] = C[2]
    y2[i-1] = D[1]
    y3 = abs((np.dot(A, C[1]) - B - Epsilon)*1e16)  # chaque élément de y3 est multiplié par 10^16 afin de rendre les erreurs plus visibles
    y4 = abs((np.dot(A, D[0]) - B))
    print("erreur avec Cholesky :", y3)
    print("erreur avec Gauss :", y4)
    tableau1.append(y3)
    tableau2.append(y4)

x = range (0, 10)
ydata1 = []
for i in range (0, 10) :
    ydata1.append(y1[i])
ydata2 = []
for i in range (0, 10) :
    ydata2.append(y2[i])
plt.plot(x, ydata1, label = "méthode de Cholesky")
plt.plot(x, ydata2, label = "méthode de Gauss")
plt.title("Evolution du nombre de calculs matriciels en fonction de la taille de la matrice")
plt.xlabel("taille de la matrice")
plt.ylabel("nombre d'opérations")
plt.grid()
plt.legend()
plt.draw()
plt.show()

print("tableau 1 :", tableau1)
print("tableau 2 :", tableau2)






