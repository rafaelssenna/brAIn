"""Memória / consolidação — M8 (Fase 2).

Mecanismo biologicamente plausível contra o esquecimento catastrófico: REPLAY,
como o replay hipocampal durante o sono. O agente guarda experiências passadas e
as REENSAIA periodicamente, impedindo que o que aprendeu (e parou de visitar)
decaia quando aprende algo novo.

A peça-chave é a AMOSTRAGEM POR RESERVATÓRIO: o buffer mantém uma amostra
*uniforme de toda a vida*, não só do passado recente. Sem isso (ex.: FIFO), as
memórias antigas seriam descartadas — exatamente o que queremos evitar.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Buffer de experiências com amostragem por reservatório (Vitter, 1985).

    Após ver N itens (N > capacidade), cada item visto tem a MESMA probabilidade
    capacidade/N de estar no buffer — uma amostra uniforme de toda a história.
    """

    def __init__(self, capacity: int, dim: int, seed: int = 0):
        self.capacity = capacity
        self.buf = np.zeros((capacity, dim), dtype=np.float64)
        self.size = 0          # quantos slots ocupados
        self.n_seen = 0        # quantos itens já passaram (para o reservatório)
        self.rng = np.random.default_rng(seed)

    def add(self, x: np.ndarray) -> None:
        if self.size < self.capacity:
            self.buf[self.size] = x
            self.size += 1
        else:
            j = int(self.rng.integers(self.n_seen + 1))   # j em [0, n_seen]
            if j < self.capacity:                          # substitui com prob cap/N
                self.buf[j] = x
        self.n_seen += 1

    def sample(self, k: int) -> np.ndarray:
        """k experiências aleatórias do buffer (com reposição)."""
        if self.size == 0:
            return np.empty((0, self.buf.shape[1]))
        idx = self.rng.integers(self.size, size=k)
        return self.buf[idx]
