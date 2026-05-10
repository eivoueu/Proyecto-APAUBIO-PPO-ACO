import pygame
import numpy as np
import torch
from src.env_casa_embrujada import CasaEmbrujada
from src.ppo import AgentePPO
from src.aco import ACO
from src.agente_hibrido import AgenteHibrido
from src.mapas import MAPA_NORMAL
    
TAM_CELDA = 80
TAM_GRID = 7
ANCHO = TAM_CELDA * TAM_GRID
ALTO = TAM_CELDA * TAM_GRID

COLOR_FONDO = (30, 30, 35)
COLOR_MURO = (100, 105, 110)
COLOR_AGENTE = (50, 150, 255)
COLOR_FANTASMA = (255, 50, 80)
COLOR_SALIDA = (50, 220, 100)
COLOR_CUADRICULA = (50, 50, 55)

def dibujar_tablero(env, pantalla):
    pantalla.fill(COLOR_FONDO)
    for x in range(TAM_GRID):
        for y in range(TAM_GRID):
            rect = pygame.Rect(y * TAM_CELDA, x * TAM_CELDA, TAM_CELDA, TAM_CELDA)
            pygame.draw.rect(pantalla, COLOR_CUADRICULA, rect, 1)
            
            if [x, y] in env.muros:
                pygame.draw.rect(pantalla, COLOR_MURO, rect)
            elif [x, y] == [TAM_GRID - 1, TAM_GRID - 1]:
                pygame.draw.rect(pantalla, COLOR_SALIDA, rect)

def visualizar_partida(muros_custom=None, nombre_mapa="Normal", id_experimento="Hibrido_Base"):
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption(f"Inferencia: {id_experimento} en {nombre_mapa}")
    reloj = pygame.time.Clock()

    env = CasaEmbrujada(tamano=TAM_GRID, muros_custom=muros_custom)
    env.fantasma_activo = True
    env.fantasma_movil = True
    
    try:
        torch.serialization.add_safe_globals([AgentePPO])
        ppo = torch.load(f"data/mejor_ppo_{nombre_mapa}_{id_experimento}.pth", weights_only=False)
        
        aco = ACO(tamano_grid=TAM_GRID)
        aco.feromonas = np.load(f"data/mejor_aco_{nombre_mapa}_{id_experimento}.npy")
        
        hibrido = AgenteHibrido(ppo, aco)
    except FileNotFoundError:
        print(f"ERROR: No se encontraron archivos para {id_experimento} en {nombre_mapa}.")
        pygame.quit()
        return

    estado, _ = env.reset()
    viejo_agente = list(env.posicion_agente)
    viejo_fantasma = list(env.posicion_fantasma)
    
    terminado = False
    truncado = False

    while not (terminado or truncado):
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                return 

        x, y = env.posicion_agente
        mascara = env.obtener_mascara_acciones()
        
        accion, _ = hibrido.elegir_accion(estado, x, y, mascara, alpha=0.3)
        estado_siguiente, recompensa, terminado, truncado, _ = env.step(accion)
        
        nuevo_agente = list(env.posicion_agente)
        nuevo_fantasma = list(env.posicion_fantasma)

        # Animación fluida entre celdas
        frames_animacion = 15
        for frame in range(frames_animacion + 1):
            progreso = frame / frames_animacion
            
            fila_agente = viejo_agente[0] + (nuevo_agente[0] - viejo_agente[0]) * progreso
            col_agente = viejo_agente[1] + (nuevo_agente[1] - viejo_agente[1]) * progreso
            
            fila_fantasma = viejo_fantasma[0] + (nuevo_fantasma[0] - viejo_fantasma[0]) * progreso
            col_fantasma = viejo_fantasma[1] + (nuevo_fantasma[1] - viejo_fantasma[1]) * progreso

            dibujar_tablero(env, pantalla)
            
            cx_agente = int(col_agente * TAM_CELDA + TAM_CELDA / 2)
            cy_agente = int(fila_agente * TAM_CELDA + TAM_CELDA / 2)
            pygame.draw.circle(pantalla, COLOR_AGENTE, (cx_agente, cy_agente), TAM_CELDA // 3)

            cx_fantasma = int(col_fantasma * TAM_CELDA + TAM_CELDA / 2)
            cy_fantasma = int(fila_fantasma * TAM_CELDA + TAM_CELDA / 2)
            pygame.draw.circle(pantalla, COLOR_FANTASMA, (cx_fantasma, cy_fantasma), TAM_CELDA // 3)

            pygame.display.flip()
            reloj.tick(60)

        estado = estado_siguiente
        viejo_agente = list(nuevo_agente)
        viejo_fantasma = list(nuevo_fantasma)
        pygame.time.delay(100)

    if recompensa > 0:
        print(">>> RESULTADO: VICTORIA <<<")
    else:
        print(">>> RESULTADO: DERROTA <<<")
        
    pygame.time.delay(1500)
    pygame.quit()

if __name__ == "__main__":
    visualizar_partida(muros_custom=MAPA_NORMAL, nombre_mapa="Normal", id_experimento="Hibrido_Base")