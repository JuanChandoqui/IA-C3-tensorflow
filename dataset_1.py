from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def leerTxt(archivo: str):
    file = f'./Datasets/{archivo}.txt'

    data = np.loadtxt(file, delimiter='\t', dtype=np.float32, skiprows=1, usecols=[0])
    data_2 = np.loadtxt(file, delimiter='\t', dtype=np.float32, skiprows=1, usecols=[1])

    array_X = []
    array_Y = []

    for value in data:
        array_X.append(value)

    for value in data_2:
        array_Y.append(value)

    print(f'ARRAY Y: {array_Y}')
    return array_X, array_Y


if __name__ == "__main__":
    #LEER TXT
    column_1 , column_2 = leerTxt('dataset01')

    # cargamos las 4 combinaciones de las compuertas XOR
    x = np.array(column_1, "float32")

    # y estos son los resultados que se obtienen, en el mismo orden
    y = np.array(column_2, "float32")

    #tasa de aprendizaje
    n = 0.1 
    
    # Create the 'Perceptron' using the Keras API
    model = tf.keras.models.Sequential()
    #inicia con pesos aleatorios con kernel_initializer
    model.add(tf.keras.layers.Dense(1, input_dim=1, activation='linear', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    
    #se utiliza adam ya que permite ajustar los pesos y sesgos, para que pueda mejorar durante el entrenamiento
    #se coloca la tasa de aprendizaje en Adam
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(n)) 

    # Train the perceptron using stochastic gradient descent
    # with a validation split of 20%
    historial = model.fit(x, y, epochs=150, batch_size=25, verbose=False)
    result = model.predict(x)
    print(f'PESOS: {model.get_weights()}')
    
    plt.plot(historial.history['loss'], label=f'n={n}')
    print(f'RESULT: {result}')
    plt.legend()
    plt.xlabel("iteraciones")
    plt.ylabel("errores")
    plt.show()