import os
import time
import numpy as np
import pandas as pd
import torch
from src.env_casa_embrujada import CasaEmbrujada
from src.ppo import AgentePPO
from src.aco import ACO
from src.agente_hibrido import AgenteHibrido

def ejecutar_experimento(alpha_max=1.0, alpha_min=0.3, muros_custom=None, nombre_mapa="Normal", 
                         lr=0.001, evaporacion=0.01, neuronas=64, id_experimento="base"):
    """
    Ejecuta el entrenamiento híbrido PPO+ACO bajo protocolo de validación estadística.
    """
    os.makedirs("data", exist_ok=True)
    semillas = [42, 101, 202, 303, 404]
    episodios = 30000
    tamano_grid = 7
    
    tiempos_detalle = []
    total_semillas = len(semillas)
    mejor_recompensa_global = -float('inf')
    
    for idx_semilla, semilla in enumerate(semillas, start=1):
        print(f"\n--- Iniciando [{id_experimento}] | Semilla {semilla} ({idx_semilla}/{total_semillas}) ---")
        t_ini = time.time()
        
        np.random.seed(semilla)
        torch.manual_seed(semilla)
        
        # Inicialización de componentes
        env = CasaEmbrujada(tamano=tamano_grid, muros_custom=muros_custom)
        ppo = AgentePPO(entradas=25, lr=lr, neuronas=neuronas)
        aco = ACO(tamano_grid=tamano_grid, evaporacion=evaporacion)
        hibrido = AgenteHibrido(ppo, aco)
        
        resultados_semilla = []
        tiempo_total_aco = 0.0
        tiempo_total_ppo = 0.0

        for episodio in range(episodios):
            # Despertar al fantasma a partir del epoch 150
            if episodio < 150:
                env.fantasma_activo = False
            else:
                env.fantasma_activo = True
                env.fantasma_movil = True
            
            if episodio < (episodios / 3):
                peso_aco = alpha_max - ((alpha_max - alpha_min) * (episodio / (episodios / 3.0)))
            else:
                peso_aco = alpha_min
                
            estado, _ = env.reset()
            terminado = False
            truncado = False
            recompensa_total = 0
            trayectoria_aco = []
            
            # Interacción agente entorno
            while not (terminado or truncado):
                x, y = env.posicion_agente
                mascara = env.obtener_mascara_acciones()
                
                accion, prob_accion = hibrido.elegir_accion(estado, x, y, mascara, alpha=peso_aco)
                estado_siguiente, recompensa, terminado, truncado, _ = env.step(accion)
                
                ppo.guardar_transicion(estado, accion, prob_accion, recompensa)
                trayectoria_aco.append((x, y, accion))
                recompensa_total += recompensa
                estado = estado_siguiente
                
            #Optimización y Diagnóstico de tiempos
            t0_ppo = time.time()
            ppo.entrenar()
            tiempo_total_ppo += (time.time() - t0_ppo)
            
            t0_aco = time.time()
            aco.evaporar()
            if recompensa_total > 0:
                aco.depositar(trayectoria_aco, recompensa_total)
            tiempo_total_aco += (time.time() - t0_aco)
                
            # Registro de métricas para el análisis
            resultados_semilla.append({
                "episodio": episodio,
                "recompensa": recompensa_total,
                "feromona_media": float(np.mean(aco.feromonas)),
                "feromona_max": float(np.max(aco.feromonas)),
                "peso_aco_aplicado": peso_aco
            })
            
            if episodio % 500 == 0:
                print(f"Episodio {episodio} | Recompensa: {recompensa_total:.2f}")
                
        df = pd.DataFrame(resultados_semilla)
        df.to_csv(f"data/resultados_{nombre_mapa}_{id_experimento}_semilla_{semilla}.csv", index=False)
        
        # Guardar el mejor modelo basado en la recompensa promedio de los últimos 100 episodios
        ultimos_100 = df['recompensa'].tail(100).mean()
        if ultimos_100 > mejor_recompensa_global:
            mejor_recompensa_global = ultimos_100
            torch.save(ppo, f"data/mejor_ppo_{nombre_mapa}_{id_experimento}.pth")
            np.save(f"data/mejor_aco_{nombre_mapa}_{id_experimento}.npy", aco.feromonas)
        
        duracion = time.time() - t_ini
        tiempos_detalle.append({
            "semilla": semilla,
            "tiempo_total": f"{int(duracion // 60)}m {int(duracion % 60)}s",
            "t_aco": f"{tiempo_total_aco:.1f}s",
            "t_ppo": f"{tiempo_total_ppo:.1f}s"
        })
        
    return tiempos_detalle

if __name__ == "__main__":
    print("\nIniciando Experimento de Control (Línea Base)")
    tiempos = ejecutar_experimento()
    for t in tiempos:
        print(f"Semilla {t['semilla']}: Total {t['tiempo_total']} | ACO: {t['t_aco']} | PPO: {t['t_ppo']}")