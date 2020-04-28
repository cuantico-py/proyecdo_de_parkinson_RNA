# creacion red neuronal para detectar parkinson! 


DESCRIPCION DEL PROBLEMA
La enfermedad de Parkinson, especialmente en sus etapas tempranas, es difícil de diagnosticar. Una de las alteraciones que se producen, debido a esta enfermedad, son cambios en la voz.
Un trabajo realizado por el profesor Rafael Orozco (profesor de la Facultad de Ingeniería de la Universidad de Antioquia), identificó características que se pueden extraer de las señales de voz de tal manera que estos descriptores sean empleados para identificar si una persona presenta o no la enfermedad.
El profesor Rafael Orozco construyó una base de datos con 50 personas sanas y 50 personas enfermas. Para esto realizó el registro de voz de cada persona e identificó 148 características o descriptores para la voz de cada paciente. Para los registros de voz cada paciente realizó un monólogo hablando sobre actividades de la vida diaria.
Los descriptores propuestos son cuatro estadísticos (mean, std, skewness, kurtosis) de: 12 MFCC y de 25
BBE (Bark Band Energies). Los valores estadísticos se calcularon teniendo en cuenta todas las grabaciones
realizadas a cada paciente.
:

la base de datos normalizada es  una base de datos normalizada y   se escogen los descriptores mas relevantes para el diagnostico tiene 11 caracteristicas que fueron las mas relevantes 


se crean 3 formas de solucionar el problema . La primera de ellas consiste en un modelo simple   1 capa oculta con 100 neuronas
 la segunda es el *Dropout* y la tercera es el *early stopping*.

Con estas tres técnicas se logra  mejorar el acierto    pasando de un modelo con precisiones del 100% y 70% en los sets de entrenamiento y validación, a casi el 75% en ambos casos.

## Dependencias
Keras==2.2.4
numpy==1.16.3
tensorflow
matploit
