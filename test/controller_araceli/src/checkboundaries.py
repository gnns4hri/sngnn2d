import numpy as np

def checkboundaries(grid, i, j, totalpuntos):
    indice = 0
    mismocluster = False
    entorno = []


##### EN EL ENTORNO TENGO EN CUENTA LOS PIXELES QUE RODEAN AL PUNTO Y ADEMAS LOS QUE RODEAN A ESTOS #####
    #### SINO DA PROBLEMAS PORQUE LOS CONSIDERA DE CLUSTERS DIFERENTES####

    for indicey in range(-2,3):
        for indicex in range(-2,3):
            entorno.append([i+indicex,j+indicey])


    for a in totalpuntos:
        for b in a:
            if (b in entorno)and (mismocluster==False):
                #print ("b esta en el entorno")
                mismocluster=True
                indice = totalpuntos.index(a)

            if (b in entorno)and(mismocluster==True):
                aux = totalpuntos.index(a)
                if (aux != indice):
                 #   print("El entorno coincide en dos listas")
                    totalpuntos[indice].extend(totalpuntos[aux])
                    totalpuntos.pop(aux)
                    break
    ### SI UN PUNTO TIENE ENTORNOS QUE PERTENECEN A DOS LISTAS
    ### CONCATENAMOS LAS DOS LISTAS Y ELIMINAMOS UNA DE ELLAS PORQUE PERTENECEN AL MISMO CLUSTER


    return mismocluster, indice