import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CasaEmbrujada(gym.Env):
    def __init__(self, tamano=7, muros_custom=None):
        super().__init__()
        self.tamano = tamano
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=4, shape=(5, 5), dtype=np.int32)
        
        self.posicion_salida = [tamano - 1, tamano - 1]
        self.estado = np.zeros((self.tamano, self.tamano), dtype=np.int32)
        self.pasos_actuales = 0
        self.max_pasos = 500
        
        self.fantasma_movil = True
        self.fantasma_activo = True
        self.fantasma_aleatorio = True
        self.historial_fantasma = []
        if muros_custom is not None:
            self.muros = muros_custom
        else:
            self.muros = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.posicion_agente = [0, 0]
        self.pasos_actuales = 0
        self.historial_fantasma = []
        
        # Aparición aleatoria del fantasma
        posiciones_validas = []
        for i in range(self.tamano):
            for j in range(self.tamano):
                pos = [i, j]
                # El fantasma no puede aparecer en un muro
                if pos in self.muros: continue
                # El fantasma no puede aparecer en la zona de inicio ni en su entorno 3x3
                if i <= 2 and j <= 2: continue
                # El fantasma no puede aparecer en la zona de salida ni en su entorno 3x3
                if i >= self.tamano - 3 and j >= self.tamano - 3: continue
                
                posiciones_validas.append(pos)
        
        # Elegir una posición aleatoria para el fantasma de las posiciones válidas
        if posiciones_validas:
            idx = np.random.choice(len(posiciones_validas))
            self.posicion_fantasma = posiciones_validas[idx]
        else:
            self.posicion_fantasma = [3, 2] 
            
        self._actualizar_matriz()
        return self._obtener_vision_local(), {}

    def _actualizar_matriz(self):
        self.estado.fill(0)
        for muro in self.muros:
            self.estado[muro[0], muro[1]] = 1
        self.estado[self.posicion_salida[0], self.posicion_salida[1]] = 3
        if self.fantasma_activo:
            self.estado[self.posicion_fantasma[0], self.posicion_fantasma[1]] = 4
        self.estado[self.posicion_agente[0], self.posicion_agente[1]] = 2

#La vision del agente es de 5x5, pero no ve al fantasma si hay un muro de por medio
    def _obtener_vision_local(self):
        vision = np.ones((5, 5), dtype=np.int32) 
        ax, ay = self.posicion_agente
        for i in range(5):
            for j in range(5):
                nx = ax - 2 + i
                ny = ay - 2 + j
                if 0 <= nx < self.tamano and 0 <= ny < self.tamano:
                    vision[i, j] = self.estado[nx, ny]
                    

        coords_fantasma = np.argwhere(vision == 4)
        if len(coords_fantasma) > 0:
            gx, gy = coords_fantasma[0]
            # Si el fantasma esta al lado, (matriz 3x3, el agente lo ve), pero si está entre el 3x3 y el 5x5, solo lo ve si no hay un muro de por medio
            if gx < 1 or gx > 3 or gy < 1 or gy > 3:
                # Mirar hacia qué dirección general está el fantasma
                dir_x = 1 if gx > 2 else (-1 if gx < 2 else 0)
                dir_y = 1 if gy > 2 else (-1 if gy < 2 else 0)
                
                # Si la casilla inmediatamente pegada al agente en esa dirección es un muro (1), no lo ve
                if vision[2 + dir_x, 2 + dir_y] == 1:
                    vision[gx, gy] = 0 
                    
        return vision
    
    def obtener_mascara_acciones(self):
        # [Arriba, Abajo, Izquierda, Derecha] -> 1.0 es libre, 0.0 es muro o límite
        mascara = np.ones(4, dtype=np.float32)
        ax, ay = self.posicion_agente
        
        # 0: Arriba (x-1)
        if ax - 1 < 0 or [ax - 1, ay] in self.muros:
            mascara[0] = 0.0
        # 1: Abajo (x+1)
        if ax + 1 >= self.tamano or [ax + 1, ay] in self.muros:
            mascara[1] = 0.0
        # 2: Izquierda (y-1)
        if ay - 1 < 0 or [ax, ay - 1] in self.muros:
            mascara[2] = 0.0
        # 3: Derecha (y+1)
        if ay + 1 >= self.tamano or [ax, ay + 1] in self.muros:
            mascara[3] = 0.0
            
        return mascara

    def _mover_fantasma(self):
        if not self.fantasma_activo: return
        
        fx, fy = self.posicion_fantasma
        ax, ay = self.posicion_agente
        
        movimientos = [
            [fx - 1, fy], [fx + 1, fy],
            [fx, fy - 1], [fx, fy + 1]
        ]
        
        mejor_distancia = float('inf')
        posibles_destinos = []
        
        for nx, ny in movimientos:
            es_valido = (0 <= nx < self.tamano) and (0 <= ny < self.tamano) and ([nx, ny] not in self.muros)
            
            # Medida para qu el fantasma no se quede esperando en una zona clave
            if es_valido and ([nx, ny] not in self.historial_fantasma):
                distancia = abs(nx - ax) + abs(ny - ay)
                
                if distancia < mejor_distancia:
                    mejor_distancia = distancia
                    posibles_destinos = [[nx, ny]]
                elif distancia == mejor_distancia:
                    posibles_destinos.append([nx, ny])
                    
        # Si tiene un camino válido, lo toma. Si está acorralado, se queda quieto (pierde el turno)
        if posibles_destinos:
            indice = np.random.choice(len(posibles_destinos))
            nueva_pos = posibles_destinos[indice]
        else:
            nueva_pos = [fx, fy] 
            
        # Actualizar posición
        self.posicion_fantasma = nueva_pos
        
        # Guardamos la decisión en memoria
        self.historial_fantasma.append(nueva_pos)
        
        # El fantasma guarda en su memoria sus ultimo 4 movimientos
        if len(self.historial_fantasma) > 4:
            self.historial_fantasma.pop(0)

    def step(self, accion):
        self.pasos_actuales += 1
        nueva_pos_agente = list(self.posicion_agente)
        
        if accion == 0: nueva_pos_agente[0] -= 1
        elif accion == 1: nueva_pos_agente[0] += 1
        elif accion == 2: nueva_pos_agente[1] -= 1
        elif accion == 3: nueva_pos_agente[1] += 1

        choco_muro = False
        if (nueva_pos_agente[0] < 0 or nueva_pos_agente[0] >= self.tamano or
            nueva_pos_agente[1] < 0 or nueva_pos_agente[1] >= self.tamano or
            nueva_pos_agente in self.muros):
            choco_muro = True
        else:
            self.posicion_agente = nueva_pos_agente

        recompensa = -0.01
        if choco_muro: recompensa -= 0.5

        terminado = False
        truncado = False
        
        if self.posicion_agente == self.posicion_salida:
            recompensa += 10.0
            terminado = True
            self._actualizar_matriz()
            return self._obtener_vision_local(), recompensa, terminado, truncado, {}
            
        if self.fantasma_activo and self.posicion_agente == self.posicion_fantasma:
            recompensa -= 10.0
            terminado = True
            self._actualizar_matriz()
            return self._obtener_vision_local(), recompensa, terminado, truncado, {}

        if self.fantasma_activo and self.fantasma_movil:
            self._mover_fantasma()
            if self.posicion_agente == self.posicion_fantasma:
                recompensa -= 10.0
                terminado = True
            
        if self.pasos_actuales >= self.max_pasos:
            truncado = True
            
        self._actualizar_matriz()
        return self._obtener_vision_local(), recompensa, terminado, truncado, {}

    def render(self):
        print(self.estado)
        print("-" * 15)