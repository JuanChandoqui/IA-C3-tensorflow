from cProfile import label
from cv2 import split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":
    # cargamos las 4 combinaciones de las compuertas XOR

    file='./Datasets/dataset03.txt'
    datax = np.loadtxt(file, delimiter=',', dtype=np.float32,  skiprows=1, usecols=[0])
    # print(datax)
    # print(datax[1])
    final_data_x = np.split(datax, 1199)
    # print(final_data_x)
    
    file='./Datasets/dataset03.txt'
    datay = np.loadtxt(file, delimiter=',', dtype=np.float64,  skiprows=1, usecols=[1])
    # print(datay)
    # print(datay[1])
    
    final_data_y = np.split(datay, 1199)

    x = np.array(datax)
      
    # y estos son los resultados que se obtienen, en el mismo orden
    y = np.array(datay)


    # Create the 'Perceptron' using the Keras API
    model = tf.keras.models.Sequential()
    #inicia con pesos aleatorios con kernel_initializer
    model.add(tf.keras.layers.Dense(1, input_dim=1, activation='linear', kernel_initializer='glorot_uniform', bias_initializer='random_uniform'))
    
    #se utiliza adam ya que permite ajustar los pesos y sesgos, para que pueda mejorar durante el entrenamiento
    #se coloca la tasa de aprendizaje en Adam
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.0001)) 
        
    # Train the perceptron using stochastic gradient descent
    # with a validation split of 20%
    historial = model.fit(x, y, epochs=100, batch_size=25, verbose=False)
    result = model.predict(x).round()

    print(f'PESOS: {model.get_weights()}')

    plt.plot(historial.history['loss'], label=f'n={0.1}')
    plt.legend()
    plt.xlabel("iteraciones")
    plt.ylabel("errores")
    plt.show()