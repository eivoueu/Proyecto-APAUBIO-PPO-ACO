import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class RedPPO(nn.Module):
    def __init__(self, entradas, num_acciones, neuronas=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(entradas * 5, neuronas),
            nn.ReLU(),
            nn.Linear(neuronas, num_acciones),
            nn.Softmax(dim=-1)
        )
        self.critico = nn.Sequential(
            nn.Linear(entradas * 5, neuronas),
            nn.ReLU(),
            nn.Linear(neuronas, 1)
        )
    
    def forward(self, x):
        x = x.long()
        x = F.one_hot(x, num_classes=5).float()
        x = x.flatten(start_dim=1)
        probs = self.actor(x)
        valor = self.critico(x)
        return probs, valor

class AgentePPO:
    def __init__(self, entradas=25, num_acciones=4, lr=0.001, neuronas=64):
        self.red = RedPPO(entradas, num_acciones, neuronas)
        self.optimizador = optim.Adam(self.red.parameters(), lr=lr)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.memoria = []

    def obtener_probabilidades(self, estado):
        estado_tensor = torch.tensor(np.array([estado]), dtype=torch.float32)
        with torch.no_grad():
            probs, _ = self.red(estado_tensor)
        return probs.numpy()[0]

    def guardar_transicion(self, estado, accion, prob_accion, recompensa):
        self.memoria.append((estado, accion, prob_accion, recompensa))

    def entrenar(self):
        if len(self.memoria) == 0:
            return
            
        estados = torch.tensor(np.array([m[0] for m in self.memoria]), dtype=torch.float32)
        acciones = torch.tensor([m[1] for m in self.memoria], dtype=torch.int64)
        probs_viejas = torch.tensor([m[2] for m in self.memoria], dtype=torch.float32)
        recompensas = [m[3] for m in self.memoria]
        
        retornos = []
        retorno_acumulado = 0
        for r in reversed(recompensas):
            retorno_acumulado = r + self.gamma * retorno_acumulado
            retornos.insert(0, retorno_acumulado)
        retornos = torch.tensor(retornos, dtype=torch.float32)
        retornos = (retornos - retornos.mean()) / (retornos.std() + 1e-7)
        
        probs_nuevas, valores_estado = self.red(estados)
        valores_estado = torch.squeeze(valores_estado)
        dist = Categorical(probs_nuevas)
        logprobs_nuevas = dist.log_prob(acciones)
        
        ventajas = retornos - valores_estado.detach()
        ratios = torch.exp(logprobs_nuevas - torch.log(probs_viejas + 1e-7))
        
        surr1 = ratios * ventajas
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * ventajas
        
        perdida_actor = -torch.min(surr1, surr2).mean()
        perdida_critico = nn.MSELoss()(valores_estado, retornos)
        perdida_total = perdida_actor + 0.5 * perdida_critico
        
        self.optimizador.zero_grad()
        perdida_total.backward()
        self.optimizador.step()
        
        self.memoria.clear()