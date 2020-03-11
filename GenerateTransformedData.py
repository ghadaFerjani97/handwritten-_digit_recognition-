import numpy as np
from scipy.linalg import qr, svd
from scipy.io import savemat
import load_DB as ldb
import PIL
import matplotlib.pyplot as plt
from random import randint
from scipy.signal import convolve2d
from  scipy.ndimage.filters import gaussian_filter
import generateSVD as ge
import DATA_matrice_moyenne as DATA

DerivsX_moyenne = np.array([deriv_translate_x(np.array(DATA.matrice_moyenne[i]).reshape(28,28)) for i in range(10)])
DerivsX_moyenne = np.array([DerivsX_moyenne[i].reshape(784) for i in range(len(DerivsX_moyenne))])

def translationX_DB(nbData=70000,nom = "",sigma = 9.):
    """
    Fonction : calcul la derive de la translation par rapport à x pour chaque image dans la BD mnist et stock le resultat  dans translate.mat
    """
    dic = {}
    dic["derivation"] = np.zeros((nbData, 784))
    i=0
    for image in range(nbData):
        data = ldb.getData(image).reshape((28,28))
        #plt.imshow(data.reshape((28, 28)))
        #plt.figure()
        #plt.imshow(data.reshape((28,28)),cmap='gray')
        #plt.show()
        
        #deriv = gaussian_filter(data,(sigma,0) );
        deriv = deriv_translate_x(data,sigma)
        #deriv = (deriv-data)/sigma
        #deriv = convolve2d(data.reshape((28, 28)), Gx, mode='same')
        dic["derivation"][i] = deriv.reshape(784)
        i+=1
    savemat("translateX"+nom, dic, do_compression=True)

def deriv_translate_x(data,sigma = 9.):
    #deriv = gaussian_filter(data,sigma);
    #deriv = (deriv - data)
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    deriv = convolve2d(data, Gx, mode='same')
    return deriv

def deriv_translate_y(data,sigma = 9.):
    #deriv = gaussian_filter(data,(0,sigma) );
    #deriv = (deriv - data)
    Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    deriv = convolve2d(data, Gy, mode='same')
    return deriv

def computeTangeant_distance(indice,alpha):
    p=np.array(DATA.matrice_moyenne[indice])
    nbp=2*alpha+1
    tp = np.array(DerivsX_moyenne[indice]).reshape(784)
    tp = np.array([tp]).transpose()
    val_alpha=np.linspace(-alpha, alpha+1,nbp,dtype=int)
    for i in range(10):
        distance=[]
        for alpha in val_alpha:
            data=translateX(p,alpha)
            e=np.array(DATA.matrice_moyenne[i])
            te = np.array(DerivsX_moyenne[i]).reshape(784)
            te = np.array([te]).transpose()
            distance.append(TangenteDistance(data,e,tp,te))
        plt.plot(val_alpha,distance)

def translationY_DB(nbData=70000,nom = "",sigma = 9):
    """
    Fonction : calcul la derive de la translation par rapport à x pour chaque image dans la BD mnist et stock le resultat  dans translate.mat
    """
    dic = {}
    dic["derivation"] = np.zeros((nbData, 784))
    i=0
    for image in range(nbData):
        data = ldb.getData(image).reshape((28,28))
        #plt.imshow(data.reshape((28, 28)))
        #plt.figure()
        #plt.imshow(data.reshape((28,28)),cmap='gray')
        #plt.show()
        deriv = deriv_translate_y(data,sigma)

        #deriv = (deriv-data)/sigma
        #deriv = convolve2d(data.reshape((28, 28)), Gx, mode='same')
        dic["derivation"][i] = deriv.reshape(784)
        i+=1
    savemat("translateY"+nom, dic, do_compression=True)

def derivRotation(data,sigma = 9):
    px = deriv_translate_x(data,sigma)
    py = deriv_translate_y(data,sigma)
    pr = np.zeros(data.shape)
    
    for x in range(28):
        for y in range(28):
            pr[x,y] = y*px[x,y]-x*py[x,y]
    return pr.reshape(784)

def Rotation_DB(nbData=70000,nom="",sigma=9):
    dic = {}
    dic["derivation"] = np.zeros((nbData, 784))
    i=0
    for image in range(nbData):
        data = ldb.getData(image).reshape((28,28))
        rot=derivRotation(data,sigma)
        dic["derivation"][i]=rot
        i+=1
    savemat("Rotation"+nom,dic,do_compression=True)
    
    
def derivScale(data):
    px = deriv_translate_x(data)
    py = deriv_translate_y(data)
    pr = np.zeros(data.shape)
    for x in range(28):
        for y in range(28):
            pr[x,y] = x*px[x,y]+y*py[x,y]
    return pr.reshape(784)

def Scale_DB(nbData=70000,nom=""):
    dic = {}
    dic["derivation"] = np.zeros((nbData, 784))
    i=0
    for image in range(nbData):
        data = ldb.getData(image).reshape((28,28))
        scale=derivScale(data)
        dic["derivation"][i]=scale
        i+=1
    savemat("Scale"+nom,dic,do_compression=True)


def derivThickening(data):
    px = deriv_translate_x(data)
    py = deriv_translate_y(data)
    pr = px*px+py*py
    return pr.reshape(784)

def Thickening_DB(nbData=70000,nom=""):
    dic = {}
    dic["derivation"] = np.zeros((nbData, 784))
    i=0
    for image in range(nbData):
        data = ldb.getData(image).reshape((28,28))
        thicken=derivThickening(data)
        dic["derivation"][i]=thicken
        i+=1
    savemat("Thickening"+nom,dic,do_compression=True)

def SVD_DB(db,nbBases = 14):
    """
    Fonction : calcul la derive de la translation par rapport à x pour chaque image dans la BD mnist et stock le resultat  dans translate.mat
    """

    dic = {}
    nbData = len(db)
    dic["data"] = np.zeros((784,10*nbBases))
    dic["label"] = np.zeros(10*nbBases, dtype=int)
    j = 0
    for i in range (10):
        U = ge.apply_svd(i, nbBases)
        dic["label"][j:j+nbBases] = i
        dic["data"][:,j:j+nbBases] = U[:,:]
        j+=nbBases

    savemat("mnist_SVD", dic, do_compression=True)


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

    lr, cr = R.shape
    lq, cq = Q.shape

    Q2 = Q[:, cr:cq]
    b = p - e
    d = Q2.transpose()@b

    return np.linalg.norm(d)

computeTangeant_distance(3,3)
#Thickening_DB()
#Scale_DB()
#Rotation_DB()
#translationX_DB()
#translationY_DB()
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
