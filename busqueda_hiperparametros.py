from entrenar import ejecutar_experimento
from src.mapas import MAPA_NORMAL
import time

print("="*60)
print("INICIANDO BARRIDO DE HIPERPARÁMETROS Y ABLACIÓN")
print("="*60)

# Estructura: (nombre_experimento, alpha_max, alpha_min, learning_rate, evaporacion, neuronas)
configuraciones = [
    # 1. Estudio de Ablación (RL simple)
    ("Ablacion_SoloPPO", 0.0, 0.0, 0.001, 0.01, 64),
    
    # 2. Experimento Base (con alpha alto y evaporacion baja)
    ("Hibrido_Base", 1.0, 0.3, 0.001, 0.01, 64),
    
    # 3. Barrido de parámetro RL
    ("Hibrido_LRAlto", 1.0, 0.3, 0.005, 0.01, 64),
    
    # 4. evaporacion alta, apha alto
    ("Hibrido_EvapAlta", 1.0, 0.3, 0.001, 0.05, 64),
    
    # 5. Alpha Bajo, Evap Baja)
    ("Hibrido_AlphaBajo", 0.5, 0.1, 0.001, 0.01, 64),           
    
    # 6. Alpha Bajo, Evap Alta
    ("Hibrido_AlphaBajo_EvapAlta", 0.5, 0.1, 0.001, 0.05, 64),   
    # 7. Barrido de parámetro Estructural (Red más grande)
    ("Hibrido_RedGrande", 1.0, 0.3, 0.001, 0.01, 128)
]

tiempo_global_inicio = time.time()

for id_exp, a_max, a_min, lr, evap, neu in configuraciones:
    print(f"\n" + "-"*40)
    print(f"EJECUTANDO CONFIGURACIÓN: {id_exp}")
    print(f"-"*40)
    
    ejecutar_experimento(
        alpha_max=a_max,
        alpha_min=a_min,
        muros_custom=MAPA_NORMAL,
        nombre_mapa="Normal",
        lr=lr,
        evaporacion=evap,
        neuronas=neu,
        id_experimento=id_exp
    )

tiempo_global_fin = time.time()
horas = int((tiempo_global_fin - tiempo_global_inicio) // 3600)
minutos = int(((tiempo_global_fin - tiempo_global_inicio) % 3600) // 60)

print("\n" + "="*60)
print(f"TIEMPO TOTAL: {horas}h {minutos}m.")
print("="*60)