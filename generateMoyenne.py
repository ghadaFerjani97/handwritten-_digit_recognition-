# -*- coding: utf-8 -*-
import load_DB as ldb
import numpy as np

Training , Test = ldb.seperateData()

somme = []
for i in range(10):
    somme.append(np.zeros(784))

nb = [0 for i in range(10)]

for e in Training:
    label = ldb.getLabel(e)
    arr = np.array(ldb.getData(e))
    somme[label] = np.add(somme[label],arr)
    nb[label]+=1
  
moyenne = []

def calcMoy(somme,nb):
    moy = []
    for e in somme:
        moy.append(int(e)//nb)
    return moy

for i in range(10):
    moyenne.append(calcMoy(somme[i],nb[i]))

#cells = np.zeros((4,4,49))
#probaCells=np.zeros(16)

    
fic = open("DATA_matrice_moyenne.py","w")
fic.write("matrice_moyenne = [\n")
for i in range(10):
    fic.write("[")
    lst = ','.join(str(e) for e in moyenne[i])
    fic.write(lst)
    fic.write("]")
    if i<9:
        fic.write(",")
    fic.write("\n")
fic.write("]\n")
fic.close()
