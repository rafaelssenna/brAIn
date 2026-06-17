"""Predictive coding HIERÁRQUICO — M9 (Fase 2).

Empilha camadas à Rao & Ballard: cada nível l prevê o nível de baixo
(r_{l-1} ≈ W_l r_l) e só o ERRO sobe pela hierarquia. As camadas altas tendem a
formar representações mais ABSTRATAS/INVARIANTES — a semente dos conceitos.

Matemática (níveis de representação r_0=x, r_1, …, r_L; pesos W_1…W_L):

  erro no nível l:   e_l = r_l - W_{l+1} r_{l+1}        (l = 0 … L-1)
  energia:           E = ½ Σ_l ‖e_l‖²   (+ prior L2 nos r ocultos)

  inferência (ajusta r):   dr_l = W_lᵀ e_{l-1} - e_l      (topo: sem o -e_l)
  aprendizado (ajusta W):  ΔW_l ∝ e_{l-1} r_lᵀ

CRUCIAL: cada ΔW_l usa só o erro logo abaixo (e_{l-1}) e a representação local
(r_l). NÃO há propagação de gradiente entre camadas — nenhum backpropagation,
nenhum "weight transport". É aqui (numa hierarquia) que o ganho do predictive
coding sobre o backprop finalmente aparece.

(Notação no código: a lista `W[i]` (i=0..L-1) liga o nível i+1 ao nível i, i.e.,
prevê r[i] a partir de r[i+1]; e[i] = r[i] - W[i] @ r[i+1].)
"""

from __future__ import annotations

import numpy as np


class HierarchicalPredictiveCoder:
    """Pilha de camadas de predictive coding. `sizes` = [D, N1, N2, …]."""

    def __init__(self, sizes, n_infer: int = 40, eta_r: float = 0.05,
                 eta_w: float = 0.02, l2_prior: float = 0.05,
                 nonneg: bool = True, seed: int = 0):
        self.sizes = list(sizes)
        self.L = len(sizes) - 1            # nº de camadas de pesos
        self.n_infer = n_infer
        self.eta_r = eta_r
        self.eta_w = eta_w
        self.l2_prior = l2_prior
        self.nonneg = nonneg

        rng = np.random.default_rng(seed)
        self.W = []
        for i in range(self.L):
            w = rng.normal(0.0, 0.1, size=(sizes[i], sizes[i + 1]))
            self.W.append(w)
        self._normalize()

    def _normalize(self):
        for i in range(self.L):
            self.W[i] = self.W[i] / (np.linalg.norm(self.W[i], axis=0, keepdims=True) + 1e-8)

    def infer(self, x: np.ndarray):
        """Relaxa as representações ocultas para minimizar a energia. Retorna a
        lista r = [x, r1, r2, …]."""
        r = [np.asarray(x, dtype=np.float64)] + \
            [np.zeros(self.sizes[i + 1]) for i in range(self.L)]
        for _ in range(self.n_infer):
            e = [r[i] - self.W[i] @ r[i + 1] for i in range(self.L)]
            for i in range(1, self.L + 1):
                dr = self.W[i - 1].T @ e[i - 1] - self.l2_prior * r[i]
                if i < self.L:                      # erro vindo de cima (existe se não é o topo)
                    dr = dr - e[i]
                r[i] = r[i] + self.eta_r * dr
                if self.nonneg:
                    r[i] = np.maximum(r[i], 0.0)
        return r

    def learn(self, x: np.ndarray):
        """Infere e aplica a regra LOCAL ΔW_l ∝ e_{l-1} r_lᵀ em cada camada."""
        r = self.infer(x)
        e = [r[i] - self.W[i] @ r[i + 1] for i in range(self.L)]
        for i in range(self.L):
            self.W[i] += self.eta_w * np.outer(e[i], r[i + 1])
        self._normalize()
        total_err = float(sum(np.sum(ei ** 2) for ei in e))
        return r, total_err

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Previsão da entrada a partir das causas inferidas (descendo a pilha)."""
        r = self.infer(x)
        pred = r[self.L]
        for i in range(self.L - 1, -1, -1):
            pred = self.W[i] @ pred
        return pred

    def top_code(self, x: np.ndarray) -> np.ndarray:
        """Representação do TOPO (a mais abstrata) para a entrada x."""
        return self.infer(x)[self.L]
