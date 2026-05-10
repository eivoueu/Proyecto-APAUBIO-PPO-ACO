# PPO + ACO Hibrido: Agente de RL en Entorno Estocastico (Casa Embrujada)

Este repositorio contiene la implementacion de un agente de Aprendizaje por Refuerzo (Reinforcement Learning) que resuelve un entorno grid-world estocastico ("Casa Embrujada"), donde un agente debe encontra la salida de un laberinto con muros, mientras que un fantasma que se mueve y aparece en posiciones aleatorias intenta detenerlo. 

El proyecto evalua una arquitectura hibrida que combina el algoritmo PPO (Proximal Policy Optimization) con un componente de memoria bioinspirado basado en ACO (Ant Colony Optimization) para mejorar la eficiencia muestral y la convergencia del agente.

## Enlace al video explicativo:

https://youtu.be/BT80TALl7ZE

## (a) Instrucciones de Instalacion

El proyecto esta desarrollado en Python. Se recomienda el uso de un entorno virtual para aislar las dependencias:

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/eivoueu/Proyecto-APAUBIO-PPO-ACO.git
   cd Proyecto-APAUBIO-PPO-ACO
   ```


2. OPCIONAL Crear y activar el entorno virtual: Por si las librerías a instalar influyen en las librerías globales del sistema, pero el código debería funcionar igualmente. 

   En Windows:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
   En Linux/Mac:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```
   (Dependencias principales: gymnasium, numpy, torch, pandas, matplotlib, pygame, scipy)

## (b) Reproducir el Experimento Principal

Para lanzar la bateria de pruebas rigurosas, el barrido de hiperparametros y el estudio de ablacion, el proyecto cuenta con un menu interactivo en consola, en el archivo "main.py", ejecutar el archivo o bien escribir el siguiente comando:

   ```bash
   python main.py
   ```

El programa empezará mostrando el siguiente menú:

==================================================

SISTEMA HÍBRIDO PPO + ACO

==================================================
1. Visualizar Mapa del laberinto
2. Ejecutar Barrido de Hiperparámetros y Ablación (Largo)
3. Generar Gráficas Estadísticas y Métricas
4. Ver Demostración (Cargar pesos y animar)
5. Salir
--------------------------------------------------
Elige una opción (1-5): 

Cada opción realiza lo siguiente:

1. "Visualizar Mapa del laberinto": Una representación visual del laberinto donde el agente (representado con un círculo azul) intentará llegar a la salida (cuadrado verde), intentando evitar un fantasma que aparece en una zona aleatoria del mapa. Esta visualización es estática solamente para entender el contexto del entorno, no hace falta entrenar el algoritmo.

2. "Entrenar y Barrido de Hiperparametros": Se entrenaran 5 diferentes semillas para 7 configuraciones diferentes en total, guardando métricas tiempos y pesos en la carpeta "/data". Además de esto, se seleccionará la mejor variante de las 5 semillas,para usarla en simulación de partidas. Como este proceso dura un poco más de una hora en total, pedirá confirmación.

3. "Generar Gráficas Estadísticas y Métricas": Muestra por consola la información obtenida del entrenamiento y barrido de hiperparámetros, además de generar en la carpeta "/figures" 6 gráficas distintas informativas y comparativas. Es necesario primero haber entrenado el modelo con la opción 2.

4. "Ver Demostración (Cargar pesos y animar)": Para ver una simulación de una partida de manera gráfica usando la mejor variante de las semillas entrenadas. El fantasma aparecerá en una casillaaleatoria y el agente debe llegar a la salida. Es necesario primero haber entrenado el modelo con la opción 2.

5. "Salir": Salir del menú iterativo terminanado el programa. Si ya se han entrenado los modelos con la opción 2, aún saliendo del programa estes se guardarán ya que los resultados están almacenados de manera permanente en /data, aunque si se vuele a entrenar el modelo con la opción 2, se volverán a calcular y entrenar los pesos de 0.

## (c) Salida Esperada y Tiempo de Ejecucion

### Tiempo de Ejecucion Aproximado
El script ejecuta un barrido completo evaluando 7 configuraciones distintas (linea base, modificaciones del algoritmo RL y variaciones del modulo bioinspirado). Para garantizar la validez estadistica, se ejecutan 5 semillas aleatorias por configuracion durante 30000 episodios.
* Tiempo total estimado: Entre 90 y 100 minutos (dependiendo del procesador).

### Salida Esperada
El pipeline de experimentacion genera tres niveles de resultados:

1. Datos Crudos (/data/): Generacion de archivos .csv con los registros metricos por episodio y exportacion de los pesos del mejor modelo de la red neuronal en formato .pth.
2. Graficas Cientificas (/figures/): Generacion de 6 paneles diagnosticos en formato .png que incluyen curvas de aprendizaje con desviacion estandar, dinamica interna del sistema adaptativo, estudio de ablacion, analisis bidimensional de sensibilidad (mapa de calor) y coste computacional.
3. Metricas en Consola: Al finalizar el procesamiento de datos, la terminal imprime un reporte con las metricas estandarizadas exigidas: puntuacion media de convergencia, eficiencia muestral (episodio de alcance del 80% del rendimiento optimo), mejora relativa frente al control y evaluacion de la significacion estadistica mediante la prueba t de Welch.

* EJEMPLO DE SALIDA DE MÉTRICAS:

============================================================
MÉTRICAS ESTÁNDAR Y RESULTADOS:
============================================================
- Convergencia Media Híbrido (Base): 4.67
- Convergencia Media Línea Base (Solo PPO): 3.05
- Mejora Relativa del Híbrido (Delta): 53.07%
- Robustez Híbrido (Inestabilidad final): 3.9691
- Eficiencia Híbrido: Alcanza 80% (7.64) en episodio 6
- Eficiencia Solo PPO: Alcanza 80% en episodio 24
- T-Test Estadística (t de Welch): p-value = 0.660060

[TIEMPOS DE EJECUCIÓN (Minutos)]
- Solo PPO: 13.62 min
- Híbrido Base: 14.63 min
- LR Alto: 10.74 min
- Evaporación Alta: 13.88 min
- Red Grande: 13.09 min
- Alpha Bajo: 12.93 min
- Alpha Bajo + Evap Alta: 14.54 min
-----------------------------------
- Tiempo Total: 93.43 min
