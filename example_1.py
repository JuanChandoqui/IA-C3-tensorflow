import numpy as np
from matplotlib import pyplot as plt
import random
import tensorflow as tf
tf.to_float = lambda x: tf.cast(x, tf.float32)

error_list = []
#Asignamos el arreglo de X & Y
X = [
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0]
    ]
    
Y = [
        [0.0],
        [0.0],
        [0.0],
        [1.0]
    ]

#print("Esto es X: ")
#print(X)
#print("Esto es Y: ")
#print(Y)

#Función de activación
def activacion(x):
  return tf.to_float(tf.greater(x, 0))

#Calculos
def calculos(X, pesos,n,peso_list):
  u = tf.matmul(X, pesos)
  salida = activacion(u) #FA
  error = tf.subtract(Y, salida) #Error
  e = tf.reduce_mean(tf.square(error)) #Error cuadratico
  #e = (np.square(Y - salida).mean()) 
  #print(e)
  error_list.append(e)
  if e <= 0.01:
    print("FIN")
    pass
  else:
    actualizar(error,pesos,X,n,peso_list) #Sigue el entrenamiento

def actualizar(error,pesos,X,n,peso_list):
  sum_pesos = tf.matmul(X, error, transpose_a=True)
  sum_pesos = (sum_pesos * n)
  pesos.assign_add(sum_pesos)
  peso_list.append(pesos)
  calculos(X,pesos,n,peso_list) #Se repite el calculo

contador = 1
for i in range(5):
  print("Generacion no ",contador)
  peso_list = []
  error_list.clear()
  n = random.random()
  #print("El valor de n es = "+str(n))
  pesos = tf.Variable(tf.random.normal([3, 1]))
  #print("Estos son los pesos generados = ",pesos)
  calculos(X,pesos,n,peso_list)
  print("peso final = ",peso_list[-1])
  plt.plot(error_list, label =f"n = {n}")
  contador +=1

#Graficación
plt.legend()
plt.ylabel('Valor del error cuadratico')
plt.xlabel('iteracion')
plt.title('Evolución del perceptrón')
plt.show()