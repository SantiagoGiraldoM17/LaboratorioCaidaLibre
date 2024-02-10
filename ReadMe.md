# Análisis de Movimiento de una Pelota con Video de Alta Velocidad

Este repositorio contiene el código, el video de muestra y las gráficas generadas de un experimento diseñado para analizar el movimiento de una pelota utilizando técnicas de visión por computadora. El experimento se lleva a cabo con un video grabado a 1000 cuadros por segundo, utilizando Python, OpenCV para el seguimiento de la pelota, y Matplotlib para generar las gráficas de distancia vs. tiempo y velocidad vs. tiempo.

## Descripción

El objetivo de este proyecto es aplicar técnicas de visión por computadora para analizar el movimiento físico de una pelota en caída libre, capturado en un video de alta velocidad. Se utiliza el video `CaidaLibreTrim.mp4` como caso de estudio para extraer datos de movimiento, calcular distancias y velocidades, y comparar los resultados experimentales con las predicciones teóricas basadas en las leyes de la física.

## Contenido del Repositorio

- `README.md`: Este archivo.
- `AnalisisCaidaLibre.py`: Script de Python que contiene el código para realizar el seguimiento de la pelota, el análisis de movimiento, y la generación de gráficas.
- `CaidaLibreTrim.mp4`: Video de muestra utilizado para el análisis.
- `grafica.png`: Gráfica generada que muestra la distancia vs. tiempo y velocidad vs. tiempo del movimiento de la pelota.
- `gms_uaj_LaboratorioCaidaLibre`: Informe de laboratorio de este experimento.

## Requisitos

Para ejecutar este proyecto, necesitarás lo siguiente:

- Python 3.x
- OpenCV library (`opencv-python`)
- Matplotlib
- Numpy

Puedes instalar las dependencias necesarias con el siguiente comando:

```bash
pip install opencv-python matplotlib numpy
```
