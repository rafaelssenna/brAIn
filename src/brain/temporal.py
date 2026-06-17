"""Previsão temporal / sequencial — M14 (Fase 3, rumo à cognição).

Até aqui o brAIn previa o INSTANTE. Aqui ele aprende a prever a PRÓXIMA
observação a partir de um CONTEXTO temporal — um traço com vazamento das entradas
recentes (memória curta, tipo um traço de elegibilidade). Aprende a estrutura
SEQUENCIAL do mundo com regra LOCAL: ΔW ∝ erro · contextoᵀ.

Por que importa: prever sequências é o substrato de "pensar à frente" (imaginar
cadeias de futuros) e a ponte natural para a LINGUAGEM, que é predição sequencial
sobre símbolos/conceitos.

  contexto:    c_t = decay·c_{t-1} + (1-decay)·x_t   (traço do passado; ordem importa)
  previsão:    x̂_{t+1} = W c_t
  erro:        ε = x_{t+1} - x̂_{t+1}
  aprende:     W += η · ε c_tᵀ      (local; sem backprop)

`decay=0` zera a memória (preditor de 1ª ordem: só o presente). `decay>0` carrega
o passado — necessário quando o próximo depende do CONTEXTO, não só do agora.
"""

from __future__ import annotations

import numpy as np


class TemporalPredictiveCoder:
    """Preditor sequencial com contexto temporal e aprendizado local."""

    def __init__(self, dim: int, decay: float = 0.5, eta_w: float = 0.05,
                 weight_decay: float = 1e-3, seed: int = 0):
        self.dim = dim
        self.decay = decay
        self.eta_w = eta_w
        self.weight_decay = weight_decay
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.1, size=(dim, dim))   # contexto -> próxima observação
        self.c = np.zeros(dim)

    def reset_context(self):
        self.c = np.zeros(self.dim)

    def predict(self) -> np.ndarray:
        """Previsão da próxima observação a partir do contexto atual."""
        return self.W @ self.c

    def observe(self, x: np.ndarray, learn: bool = True):
        """Recebe a observação real x (a 'próxima'); compara com a previsão feita a
        partir do contexto, aprende, e atualiza o contexto. Retorna (erro, previsão)."""
        pred = self.W @ self.c
        err = x - pred
        if learn:
            self.W += self.eta_w * (np.outer(err, self.c) - self.weight_decay * self.W)
        self.c = self.decay * self.c + (1.0 - self.decay) * x   # atualiza memória
        return float(np.sum(err ** 2)), pred

    def generate(self, n: int, prototypes: np.ndarray):
        """IMAGINA a continuação: prevê o próximo, encaixa no protótipo mais
        parecido, realimenta como entrada, repete. Retorna os índices gerados."""
        out = []
        protos = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9)
        for _ in range(n):
            pred = self.W @ self.c
            idx = int(np.argmax(protos @ pred))        # protótipo mais próximo
            out.append(idx)
            x = prototypes[idx]
            self.c = self.decay * self.c + (1.0 - self.decay) * x
        return out
