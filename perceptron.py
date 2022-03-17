from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import random
import math

def perceptron(w,x,yd,n,umbral):
    errores = []
    error_actual = 1
    yc = []
    k = 0
    while abs(error_actual) > umbral:
        # print(f"---------------k{k}------------------")
        # print(f'w{k}={w}')
        u = np.dot(x,w)
        yc = u
        ek = yd-yc
        nek = np.array([n*valor for valor in np.dot(ek,x)])
        w = w + nek
        error_actual = abs(np.sum(ek))
        errores.append(error_actual)
        error_actual = round(np.sum(np.array([e**2 for e in ek]))**0.5,3)
        errores.append(error_actual)
        k+=1
        # print(f'u={u}')
        # print(f'yc={yc}')
        # print(f'ek={ek}')
        # print(f'nek={nek}')
        # print(f'w{k}={w}')
        # print(f'm={error_actual}')
        # sleep(3)
    return yc,w,errores

def graficar(errores):
    for error in errores:
        label = f"n={error[1]}"
        plt.plot(error[0], label=label)
    plt.legend()
    plt.xlabel("Iteraciones (k)")
    plt.ylabel("Errores")
    plt.title("")
    plt.show()

def main_perceptron(x,yd,w0,tazas):
    data = []
    # x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    umbral = 0.04
    errores = []
    for n in tazas:
        yc, w, error = perceptron(w0,x,yd,n,umbral)
        print(f'--------------taza de aprendizaje= {n}------------------\nsalida esperada = {yd} \nsalida obtenida = {yc}')
        print(f"Pesos: {w}")
        print(f"Iteraciones(K):{len(error)}")
        # print(f"Errores: {error}")
        errores.append((error,n))
        data.append((w,len(error),n))
    graficar(errores)
    return data

if __name__ =='__main__':
    with open('Datasets/dataset01(1).txt', 'r') as f:
        lineas = [linea.split() for linea in f]

    x = []
    y = []
    for linea in lineas:
        x.append([0.00001,float(linea[0])])
        y.append(float(linea[1]))

    x2 = np.array(x)
    y2 = np.array(y)
    print(x2)

    w0 = np.array([random.uniform(0,1) for _ in range(2)])
    tasas_aprendizaje = [0.0000002,0.0000001,0.0000005,0.00000015,0.00000008]
    print(f'pesos iniciales = {w0}')
    main_perceptron(x2,y2,w0,tasas_aprendizaje)