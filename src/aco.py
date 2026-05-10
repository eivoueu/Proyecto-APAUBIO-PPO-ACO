import numpy as np

class ACO:
    def __init__(self, tamano_grid, num_acciones=4, evaporacion=0.01, feromona_inicial=1.0):
        self.tamano_grid = tamano_grid
        self.num_acciones = num_acciones
        self.evaporacion = evaporacion
        self.tau_min = 0.1
        self.tau_max = 10.0
        
        self.feromonas = np.full((tamano_grid, tamano_grid, num_acciones), feromona_inicial, dtype=np.float32)

    def obtener_feromonas(self, x, y):
        return self.feromonas[x, y]

    def evaporar(self):
        self.feromonas = (1.0 - self.evaporacion) * self.feromonas
        self.feromonas = np.clip(self.feromonas, self.tau_min, self.tau_max)

    def depositar(self, trayectoria, recompensa_total):
        if recompensa_total > 0:
            Q = 10.0 
            cantidad_deposito = Q / len(trayectoria) 
            
            for (x, y, accion) in trayectoria:
                self.feromonas[x, y, accion] += cantidad_deposito
                
            self.feromonas = np.clip(self.feromonas, self.tau_min, self.tau_max)