#--coding: utf-8--
import numpy as np
import load_DB as ldb


"""
Entrees: U := la bases des u_i d'un chiffre, image := l'indice d'un image non connu 
Sortie: la distance minimal de l'image au plan Vect(U)
"""
def distance_de_base(label,image,M):
    v = ldb.getData(image)
    Mv = M[label].dot(v)# multiplication de M*v
    return np.linalg.norm(Mv)

