
# En este proyecto se implementan 3 redes neuronales para la solucion de un mismo problema
# por: Johan Esteban Moreno Zapata
# Importación de librerías
# debes tener instalado las siguientes librerias
#numpy, matplotlib, keras, tensorflow, sklearn

import numpy as np
import matplotlib.pyplot as plt


# Pre-procesamiento de datos
from sklearn.preprocessing import StandardScaler # para normalizar datos
from sklearn.model_selection import train_test_split

# Modelos en Keras (incluyendo Droput y Early Stopping)
from keras.models import Sequential
from keras.layers import Dense, Dropout  # librerias paras ANR  de KERAS
from keras.callbacks import EarlyStopping  #keras corre sobre tensorflow para correr en GPU

# Función para graficar resultados
#
def graficar_resultados(historia):
	plt.subplot(1,2,1)
	plt.plot(historia.history['loss'])
	plt.plot(historia.history['val_loss'])
	plt.ylabel('error')
	plt.xlabel('Iteración')
	plt.legend(['Entrenamiento','Validación'])

	plt.subplot(1,2,2)
	plt.plot(historia.history['accuracy'])
	plt.plot(historia.history['val_accuracy'])
	plt.ylabel('Precisión')
	plt.xlabel('Iteración')
	plt.legend(['Entrenamiento','Validación'])

	ax = plt.gca()
	ax.yaxis.set_label_position("right")
	ax.yaxis.tick_right()

	plt.show()

#
# Lectura y pre-procesamiento de datos
#
dataset = np.loadtxt("data2.csv", delimiter=",")
X = dataset[:, 0:11]   # entrada de datos son 11 caracteristicas
Y = dataset[:, 11] # salida esperada de datos, esta en la columna 12 ( vector de 11 pocisiones)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X) # estas tres lineas normalizan los datos de entrada

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# divido los datos en entrenamiento y validacion 20% para test  el resto para entrenamiento
#
# Creación, entrenamiento y validación de los modelos
#

np.random.seed(100)



# 1. Modelo simplificado: 1 capa oculta con 100 neuronas
model = Sequential() # modelo secuencial donde hay capas de neuronas y le paso una lista
model.add(Dense(100, input_dim=11, activation='relu')) # primer capa densa  de 100 neuronas y tiene 11
# entradas de las caracteristicas  y elijo una funcion de activacion, en este caso una "RELU"
model.add(Dense(1, activation='sigmoid')) # capa intermedia,  Dense ya sabe que me debe recibir todas las salidas de
# la capa anterior, con 1 neurona de salida  y la funcion de activacion es una sigmoide que me da valores de 0 y 1
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# compilo los datos y aggrego# funcion de perdida(binary_crossentropy), esta funcion es excelente para salidas de 0 o 1
# optmizador( adam) de las mejores funciones para ARN
# y metrica de desempeño quiero que cuando compile me saque unas metricas en este caso % de  aciertos

historia = model.fit(x_train, y_train, batch_size=64, epochs=150, validation_data=(x_test,y_test), verbose=1)

graficar_resultados(historia)

# 3. Modelo base con dropout
model = Sequential()
model.add(Dense(1000, input_dim=11, activation='relu'))
model.add(Dropout(0.9)) # solo agrego esta funcion para apagar aleatoriamente ciertas neuronas mejora el desempeño
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
historia = model.fit(x_train, y_train, batch_size=64, epochs=150, validation_data=(x_test,y_test), verbose=1)

graficar_resultados(historia)

# 4. Modelo con droput y early stopping
# solo es parar cuando se empieza aumnetar el error de validacion y en este pto me detengo
model = Sequential()
model.add(Dense(1000, input_dim=11, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
historia = model.fit(x_train, y_train, batch_size=64, epochs=150, validation_data=(x_test,y_test), verbose=1, callbacks=[early_stop])
#model.fit  corre el entrenamiento con los datos de entrada y salida y calcula la funcion adecuada
# agarrar el set completo de datos recorrerlo 150 epocas
# btach size le digo cada cuanto ajusto los pesos
graficar_resultados(historia)
