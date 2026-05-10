import sys
from src.mapas import MAPAS, MAPA_NORMAL
from entrenar import ejecutar_experimento
from visualizar_laberinto import run_visualizer
from visualizar_partida import visualizar_partida
from plot_results import plotear_resultados
import time
import pandas as pd

def menu_principal():
    mapa_actual = MAPA_NORMAL
    nombre_mapa_actual = "Normal"

    while True:
        print("\n" + "="*50)
        print("SISTEMA HÍBRIDO PPO + ACO")
        print("="*50)
        print("1. Visualizar Mapa del laberinto")
        print("2. Ejecutar Barrido de Hiperparámetros y Ablación (Largo)")
        print("3. Generar Gráficas Estadísticas y Métricas")
        print("4. Ver Demostración (Cargar pesos y animar)")
        print("5. Salir")
        print("-"*50)
        
        opcion = input("Elige una opción (1-5): ")
                
        if opcion == '1':
            print(f"\nVisualizar Mapa del laberinto: {nombre_mapa_actual}")
            run_visualizer(muros_custom=mapa_actual, nombre_mapa=nombre_mapa_actual)
            
        elif opcion == '2':
                configuraciones = [
                    ("Ablacion_SoloPPO", 0.0, 0.0, 0.001, 0.01, 64),
                    ("Hibrido_Base", 1.0, 0.3, 0.001, 0.01, 64),       
                    ("Hibrido_LRAlto", 1.0, 0.3, 0.005, 0.01, 64),
                    ("Hibrido_EvapAlta", 1.0, 0.3, 0.001, 0.05, 64),   
                    ("Hibrido_RedGrande", 1.0, 0.3, 0.001, 0.01, 128),
                    ("Hibrido_AlphaBajo", 0.5, 0.1, 0.001, 0.01, 64),           
                    ("Hibrido_AlphaBajo_EvapAlta", 0.5, 0.1, 0.001, 0.05, 64)   
                ]
                
                tiempos_ejecucion = []
                
                for id_exp, a_max, a_min, lr, evap, neu in configuraciones:
                    print(f"\nProcesando: {id_exp}")
                    t_inicio = time.time()
                    
                    ejecutar_experimento(
                        alpha_max=a_max, alpha_min=a_min,
                        muros_custom=mapa_actual, nombre_mapa=nombre_mapa_actual,
                        lr=lr, evaporacion=evap, neuronas=neu,
                        id_experimento=id_exp
                    )
                    
                    t_fin = time.time()
                    minutos = (t_fin - t_inicio) / 60.0
                    tiempos_ejecucion.append({"ID": id_exp, "Tiempo_Minutos": minutos})
                pd.DataFrame(tiempos_ejecucion).to_csv("data/tiempos_ejecucion.csv", index=False)

        elif opcion == '3':
            try:
                plotear_resultados()
            except Exception as e:
                print(f"Error al generar gráficas: {e}")
                print("Asegúrese de haber ejecutado la opción 2 primero.")
                
        elif opcion == '4':
            print(f"\nAbriendo simulación animada en el mapa {nombre_mapa_actual}")
            visualizar_partida(muros_custom=mapa_actual, nombre_mapa=nombre_mapa_actual)
        
        elif opcion == '5':
            sys.exit()
            
        else:
            print("\nOpción no reconocida. Por favor, elige un número del 1 al 5.")

if __name__ == "__main__":
    menu_principal()