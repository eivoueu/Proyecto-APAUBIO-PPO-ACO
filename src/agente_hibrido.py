import numpy as np

class AgenteHibrido:
    def __init__(self, ppo, aco):
        self.ppo = ppo
        self.aco = aco
        
    def elegir_accion(self, estado, x, y, mascara_acciones, alpha=1.0):
        probs_ppo = self.ppo.obtener_probabilidades(estado)
        feromonas = self.aco.obtener_feromonas(x, y)
        
        # Si el agente ve al fantasma, se le da más peso al PPO
        if 4 in estado:
            alpha_dinamico = 0.15  
        else:
            alpha_dinamico = alpha

        feromonas_pesadas = feromonas ** alpha_dinamico
        
        probabilidades_brutas = probs_ppo * feromonas_pesadas * mascara_acciones
        suma = np.sum(probabilidades_brutas)
        suma_mascara = np.sum(mascara_acciones)
        
        if suma == 0:
            if suma_mascara > 0:
                probs_finales = mascara_acciones / suma_mascara
            else:
                probs_finales = np.ones(4) / 4.0
        else:
            probs_finales = probabilidades_brutas / suma
            
        accion_elegida = np.random.choice(4, p=probs_finales)
        prob_accion_ppo = probs_ppo[accion_elegida]
        
        return accion_elegida, prob_accion_ppo