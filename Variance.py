#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 03:34:32 2019

@author: user
"""
import load_DB as ldb
import numpy as np
Training , Test = ldb.seperateData()

somme = []
somme2 = []
for i in range(10):
    somme.append(np.zeros(784))
    somme2.append(np.zeros(784))


nb = [0 for i in range(10)]

for e in Training:
    label = ldb.getLabel(e)
    arr = np.array(ldb.getData(e))
    somme[label] = np.add(somme[label],arr)
    nb[label]+=1
  
moyenne = []
variance = []

def calcMoy(somme,nb):
    moy = []
    for e in somme:
        moy.append(int(e)//nb)
    return moy

for i in range(10):
    moyenne.append(calcMoy(somme[i],nb[i]))

moyenne=np.array(moyenne)
for e in Training:
    label = ldb.getLabel(e)
    arr = np.array(ldb.getData(e))
    somme2[label] = np.add(somme2[label],(arr-moyenne[label])**2)
    nb[label]+=1

def calcVar(somme,nb):
    var = []
    for e in somme:
        var.append(int(e)//nb)
    return var

for i in range(10):
    variance.append(calcVar(somme2[i],nb[i]))

variance = np.sqrt(variance)
variance.astype(int)



#cells = np.zeros((4,4,49))
#probaCells=np.zeros(16)

#    
fic = open("DATA_matrice_variance.py","w")
fic.write("matrice_variance = [\n")
for i in range(10):
    fic.write("[")
    lst = ','.join(str(e) for e in variance[i])
    fic.write(lst)
    fic.write("]")
    if i<9:
        fic.write(",")
    fic.write("\n")
fic.write("]\n")
fic.close()
