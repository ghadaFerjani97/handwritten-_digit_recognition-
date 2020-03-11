# -*- coding: utf-8 -*-
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
# chargement de la base de donnée
global mnist
mnist = loadmat("mnist-original.mat")
global mnist_data
mnist_data = mnist["data"].T
global mnist_label
mnist_label = mnist["label"][0]
nbChiffre = [6903, 7877, 6990, 7141, 6824, 6313, 6876, 7293, 6825, 6958]


def resetDataBase(database_file_name):
    """
    remplace la BD initial par le DB database_file_name
    """
    print("RESET DATA ...")
    global mnist_data
    mnist = loadmat(database_file_name)
    mnist_data = mnist["data"].T
    global mnist_label
    mnist_label = mnist["label"][0]

def getDerivationDB(db):
    """
    retourne un vecteur composé de tous les derivations possibles
    """
    db = loadmat(db)
    return np.array(db["derivation"])

# retourne une liste d'indices de tous les chiffres n dans la bese de donnée mnist
def findChiffre(n):
    lst = []
    for i in range(len(mnist_data)):
        if int(mnist_label[i]) == n:
            lst.append(i)
    return lst


# retourne une liste des indices des chiffres n dans la listedb
def findChiffre_liste(n, db):
    lst = []
    for i in range(len(db)):
        if int(mnist_label[db[i]]) == n:
            lst.append(i)
    return lst


# retourne les pixel du chiffre de l'indice donné sous forme d'une liste
def getData(indice):
    if indice >= 0 and indice < len(mnist_data):
        return np.array(mnist_data[indice])


# affiche le chiffre de l'indice donné
def afficheChiffre(indice):
    plt.figure()
    img = np.array(getData(indice))
    img = img.reshape((28,28)).transpose()
    img = 255*np.ones(img.shape)-img
    plt.imshow(img)


# retourne le label du chiffre à l'indice donné
def getLabel(indice):
    if indice >= 0 and indice < len(mnist_label):
        return int(mnist_label[indice])


# retourne un liste d'indices pour les donnés d'entrainement et une liste d'indice pour les donnés de test
def seperateData(ratio=0.8):
    Training = []
    Test = []
    lst = [findChiffre(i) for i in range(10)]
    for n in range(10):
        limit = int(ratio * nbChiffre[n])
        for i in range(nbChiffre[n]):
            if i < limit:
                Training.append(lst[n][i])
            else:
                Test.append(lst[n][i])
    Test = random.sample(Test,len(Test))
    return Training,Test

