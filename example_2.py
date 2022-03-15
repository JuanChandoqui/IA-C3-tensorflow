from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":
    # cargamos las 4 combinaciones de las compuertas XOR
    x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]], "float32")
    
    tasas_aprendizaje = [0.12, 0.51, 0.61, 0.51, 0.82]

    # y estos son los resultados que se obtienen, en el mismo orden
    y = np.array([[1],[0],[1],[0]], "float32")

    # Create the 'Perceptron' using the Keras API
    model = tf.keras.models.Sequential()
    #inicia con pesos aleatorios con kernel_initializer
    model.add(tf.keras.layers.Dense(1, input_dim=3, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    
    for value in tasas_aprendizaje:
        #se utiliza adam ya que permite ajustar los pesos y sesgos, para que pueda mejorar durante el entrenamiento
        #se coloca la tasa de aprendizaje en Adam
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(value)) 
        
        # Train the perceptron using stochastic gradient descent
        # with a validation split of 20%
        historial = model.fit(x, y, epochs=100, verbose=False)
        result = model.predict(x).round()
        # print (result)
        print(model.get_weights())
        # print(historial.history['loss'])
        plt.plot(historial.history['loss'], label=f'n={value}')
    plt.legend()
    plt.xlabel("iteraciones")
    plt.ylabel("errores")
    plt.show()