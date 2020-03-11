import numpy as np
from scipy.linalg import qr, svd
from scipy.io import savemat
import load_DB as ldb
import PIL
import matplotlib.pyplot as plt
from random import randint
from scipy.signal import convolve2d

import generateSVD as ge

def translationX_DB(db = None,nom = ""):
    """
    Fonction : calcul la derive de la translation par rapport à x pour chaque image dans la BD mnist et stock le resultat  dans translate.mat
    """
    dic = {}
    if db==None:
        db = np.linspace(0,70000,1,dtype=int)
    nbData = len(db)
    dic["derivation"] = np.zeros((nbData, 784))
    i = 0
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    for image in db :
        data = ldb.getData(image)
        deriv = convolve2d(data.reshape((28, 28)), Gx, mode='same')
        dic["derivation"][i] = deriv.reshape(784)
        i += 1
    savemat("translateX"+nom, dic, do_compression=True)

def SVD_DB(db,nbBases = 20):
    """
    Fonction : calcul la derive de la translation par rapport à x pour chaque image dans la BD mnist et stock le resultat  dans translate.mat
    """
    dic = {}
    nbData = len(db)
    dic["svd"] = []
    dic["label"] = []
    for i in range (10):
        U = ge.apply_svd(i, nbBases)
        U = U.transpose()
        for ligne in U :
            dic["svd"].append(ligne)
            dic["label"].append(i)

    savemat("mnist_SVD", dic, do_compression=True)

def translationY_DB():
    """
    Fonction : calcul la derive de la translation par rapport à y pour chaque image dans la BD mnist et stock le resultat  dans translate.mat
    """
    dic = {}
    nbData = 70000
    dic["derivation"] = np.zeros((nbData, 784))
    i = 0
    Gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).transpose()
    for image in range(nbData):
        data = ldb.getData(image)
        deriv = convolve2d(data.reshape((28, 28)), Gy, mode='same')
        dic["derivation"][i] = deriv.reshape(784)
        i += 1
    savemat("translateY", dic, do_compression=True)


def translateX(image, alphaX):
    """
    translateX : (image : 1x784, alphaX : int  ) ==> imageTranslated, T:differentielle de l'operation
    Fonction : Translate l'image 'image' de alphaX suivant l'Horizontal
    """
    image = image.reshape((28, 28))
    imageTranslated = np.zeros(image.shape, dtype="int32")
    if alphaX >= 0:
        imageTranslated[:, alphaX:image.shape[1]] = image[:, 0:(image.shape[1] - alphaX)]
    else:
        image = np.flip(image, (0, 1))
        alphaX = -alphaX
        imageTranslated[:, alphaX:image.shape[1]] = image[:, 0:(image.shape[1] - alphaX)]
        imageTranslated = np.flip(imageTranslated, (0, 1))
    return imageTranslated.reshape(784)


def translateY(image, alphaY):
    """
    translateY : (image : 1x784, alphaX : int  ) ==> imageTranslated, T:differentielle de l'operation
    Fonction : Translate l'image 'image' de alphaY suivant la verticale
    """
    image = image.reshape((28, 28))
    imageTranslated = np.zeros(image.shape, dtype="int32")
    if alphaY >= 0:
        imageTranslated[alphaY:image.shape[1], :] = image[0:(image.shape[1] - alphaY), :]
    else:
        image = np.flip(image, (0, 1))
        alphaY = -alphaY
        imageTranslated[alphaY:image.shape[1], :] = image[0:(image.shape[1] - alphaY), :]
        imageTranslated = np.flip(imageTranslated, (0, 1))
    return imageTranslated.reshape(784)


def TangenteDistance(p, e, Tp, Te):
    """
    Fonction calcul la distance tangente entre deux images p,e
    Entrées: p,e: deux images,  Tp: la matrice des transormations de p, Te: la matrice des transformations des e
    Tp et Te ont la même dimension
    Sortie: d: la distance tangente de p et e
    """
    lp, cp = Tp.shape
    le, ce = Te.shape
    A = np.zeros((lp, cp + ce))
    A[:, 0:cp] = -1 * Tp[:, :]
    A[:, cp:ce + cp] = Te[:, :]
    Q, R = qr(A)
    #    print(A.shape)
    # U,S,V = svd(A)
    #    print(U)
    #    print(S)
    # U2 = U[:,len(S):U.shape[0]]

    # print(Q.shape,"\n\n")
    #    #print(R.shape)
    lr, cr = R.shape
    lq, cq = Q.shape
    Q2 = Q[:, cr:cq]
    b = p - e
    # b2 = b[len(S):U.shape[0]]
    d = Q2.transpose() @ b
    # d = U2.transpose()@b2
    return np.linalg.norm(d)

# translationX_DB()
# Training, Test = ldb.seperateData()
# alpha = 4
##ldb.resetDataBase("translateX.mat")
# derivs = ldb.getDerivationDB("translateX.mat")
# p = ldb.getData(15)
# tp = np.array([derivs[15]]).transpose()
# e = ldb.getData(20)
# te = np.array([derivs[20]]).transpose()
# print(TangenteDistance(p,e,tp,te))
# print(np.linalg.norm(e-p))
# ldb.afficheChiffre(ldb.getData(15))
# ldb.afficheChiffre(ldb.getData(20))
# M = ldb.getData(10)
# trX = translateX(M,5)
# trY = translateY(M,5)
##translation(trainingData,0)
##img = translateX(M,1)
# plt.imshow(M.reshape(28,28))
# plt.show()
# plt.imshow(trX)
# plt.show()
# plt.imshow(trY)
# plt.show()