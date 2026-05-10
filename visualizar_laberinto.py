import pygame
import numpy as np
from src.env_casa_embrujada import CasaEmbrujada
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

def run_visualizer(muros_custom=None, nombre_mapa="Normal"):
    env = CasaEmbrujada(tamano=TAM_GRID, muros_custom=muros_custom)
    env.fantasma_activo = True 
    env.reset() 
    
    matriz_laberinto = env.estado 

    pygame.init()
    pygame.display.set_caption(f"Nombre del mapa: {nombre_mapa.upper()}")
    screen = pygame.display.set_mode((ANCHO, ALTO))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill(COLOR_FONDO)

        for fila in range(TAM_GRID):
            for columna in range(TAM_GRID):
                valor = matriz_laberinto[fila, columna]
                rect = pygame.Rect(columna * TAM_CELDA, fila * TAM_CELDA, TAM_CELDA, TAM_CELDA)

                pygame.draw.rect(screen, COLOR_CUADRICULA, rect, 1)

                if valor == 1:
                    pygame.draw.rect(screen, COLOR_MURO, rect)
                elif valor == 3:
                    pygame.draw.rect(screen, COLOR_SALIDA, rect)
                elif valor == 2:
                    centro = (columna * TAM_CELDA + TAM_CELDA // 2, fila * TAM_CELDA + TAM_CELDA // 2)
                    pygame.draw.circle(screen, COLOR_AGENTE, centro, TAM_CELDA // 3)
                elif valor == 4:
                    centro = (columna * TAM_CELDA + TAM_CELDA // 2, fila * TAM_CELDA + TAM_CELDA // 2)
                    pygame.draw.circle(screen, COLOR_FANTASMA, centro, TAM_CELDA // 3)

        pygame.display.flip()
        clock.tick(30) 

    pygame.quit()
    return

if __name__ == "__main__":
    run_visualizer(muros_custom=MAPA_NORMAL, nombre_mapa="Normal")