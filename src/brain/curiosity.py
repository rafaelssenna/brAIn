"""Motivação intrínseca / curiosidade — M5.

Implementa o sinal de "learning progress" (Oudeyer): a curiosidade não busca
onde o erro é MAIOR (isso prenderia o agente numa "TV chiando" — ruído puro, que
nunca para de surpreender), e sim onde o erro está CAINDO MAIS RÁPIDO — ou seja,
onde o agente está de fato APRENDENDO.

Para cada região/atividade, mantém duas médias exponenciais do erro de previsão:
uma rápida e uma lenta. O progresso é o quanto a lenta supera a rápida:

    LP_k = max(0, EMA_lenta_k - EMA_rápida_k)

Erro caindo  => lenta > rápida => LP > 0 (está aprendendo).
Erro estável => lenta ≈ rápida => LP ≈ 0 (já dominou, OU é ruído inaprendível).

Assim o agente abandona tanto o que já domina quanto o que é impossível, e gasta
sua atenção onde o aprendizado rende — o que faz emergir um "currículo" natural.
"""

from __future__ import annotations

import numpy as np


class IntrinsicMotivation:
    """Rastreia o learning progress por região (estimador de curiosidade)."""

    def __init__(self, n_regions: int, tau_fast: float = 15.0, tau_slow: float = 80.0):
        self.n_regions = n_regions
        self.alpha_fast = 1.0 / tau_fast
        self.alpha_slow = 1.0 / tau_slow
        self.e_fast = np.full(n_regions, np.nan)   # EMA rápida do erro
        self.e_slow = np.full(n_regions, np.nan)   # EMA lenta do erro
        self.visits = np.zeros(n_regions, dtype=int)

    def update(self, region: int, error: float) -> None:
        """Registra um novo erro de previsão observado na região."""
        if np.isnan(self.e_fast[region]):          # primeira visita: inicializa
            self.e_fast[region] = error
            self.e_slow[region] = error
        else:
            self.e_fast[region] += self.alpha_fast * (error - self.e_fast[region])
            self.e_slow[region] += self.alpha_slow * (error - self.e_slow[region])
        self.visits[region] += 1

    def learning_progress(self) -> np.ndarray:
        """LP por região: max(0, EMA_lenta - EMA_rápida). Região nunca visitada
        recebe LP infinito (otimismo: vale a pena ir ver)."""
        lp = self.e_slow - self.e_fast
        lp = np.where(np.isnan(lp), np.inf, np.maximum(0.0, lp))
        return lp

    def current_error(self) -> np.ndarray:
        """EMA rápida do erro por região (nan se nunca visitada)."""
        return self.e_fast.copy()


def select_region(motivation: IntrinsicMotivation, policy: str,
                  rng: np.random.Generator, eps: float = 0.1) -> int:
    """Escolhe a próxima região segundo a política de exploração.

    - 'curiosity': maior learning progress (ε-greedy para seguir estimando).
    - 'novelty'  : maior erro atual (a ingênua — cai na armadilha da TV chiando).
    - 'random'   : uniforme (baseline).
    """
    n = motivation.n_regions
    if policy == "random":
        return int(rng.integers(n))
    if policy == "novelty":
        err = motivation.current_error()
        if np.any(np.isnan(err)):                  # visita não-explorados primeiro
            return int(np.flatnonzero(np.isnan(err))[0])
        return int(np.argmax(err))
    if policy == "curiosity":
        if rng.random() < eps:                     # exploração ε-greedy
            return int(rng.integers(n))
        lp = motivation.learning_progress()
        return int(np.argmax(lp))                   # inf (não-visitado) vence => explora
    raise ValueError(f"política desconhecida: {policy}")
