# creacion red neuronal para detectar parkinson! 

la base de datos  (data2csv) es la misma base de datos usada en el codigo de matlab, esta base de datos ya esta normalizada y solo tiene 11 caracteristicas que fueron las mas relevantes 


se crean 3 formas de solucionar el problema . La primera de ellas consiste en un modelo simple   1 capa oculta con 100 neuronas
 la segunda es el *Dropout* y la tercera es el *early stopping*.

Con estas tres técnicas se logra  mejorar el acierto    pasando de un modelo con precisiones del 100% y 70% en los sets de entrenamiento y validación, a casi el 75% en ambos casos.

## Dependencias
Keras==2.2.4
numpy==1.16.3