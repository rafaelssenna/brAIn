"""Organismo integrado — M7 (Fase 2).

Costura os ingredientes da Fase 1 num ÚNICO agente:
  • CORPO (M3): vive num mundo espacial (anel de lugares) e precisa SE MOVER.
  • MODELO DE MUNDO (M4): um predictive coder aprende, online e local, os
    padrões sensoriais de cada lugar.
  • CURIOSIDADE (M5): learning progress por lugar guia PARA ONDE o corpo navega.

Active inference em espírito: perceber minimiza surpresa; agir busca o lugar de
maior progresso de aprendizado — mas agora a "atenção" é FÍSICA: custa passos
chegar lá. O corpo torna a alocação de atenção consequente.

Substrato: rate-based (reusa M4/M5) — a versão spiking vem no M10.
"""

from __future__ import annotations

import numpy as np

from .predictive import PredictiveCoder
from .curiosity import IntrinsicMotivation
from .memory import ReplayBuffer


class RingWorld:
    """Anel de N lugares; cada lugar emite um padrão sensorial fixo (aprendível)
    ou ruído puro (inaprendível). Mover-se é em passos pelo anel."""

    def __init__(self, n_locations: int = 8, grid: int = 8,
                 noise_locs=(3, 6), seed: int = 0):
        self.n = n_locations
        self.grid = grid
        self.D = grid * grid
        self.noise_locs = set(noise_locs)

        pool = self._make_bars(grid)          # padrões distintos disponíveis
        self.patterns = {}
        li = 0
        for loc in range(n_locations):
            if loc in self.noise_locs:
                self.patterns[loc] = None     # ruído
            else:
                self.patterns[loc] = pool[li % len(pool)]
                li += 1

    @staticmethod
    def _make_bars(grid):
        pats = []
        for r in range(grid):
            p = np.zeros((grid, grid)); p[r, :] = 1.0; pats.append(p.ravel())
        for c in range(grid):
            p = np.zeros((grid, grid)); p[:, c] = 1.0; pats.append(p.ravel())
        P = np.array(pats)
        return P / np.linalg.norm(P, axis=1, keepdims=True)

    def sense(self, pos: int, rng, noise: float = 0.05) -> np.ndarray:
        p = self.patterns[pos]
        if p is None:                          # ruído: vetor novo a cada visita
            x = rng.standard_normal(self.D)
            return x / np.linalg.norm(x)
        x = p + noise * rng.standard_normal(self.D)
        return x / np.linalg.norm(x)

    def is_noise(self, pos: int) -> bool:
        return pos in self.noise_locs

    def learnable_locs(self):
        return [l for l in range(self.n) if l not in self.noise_locs]


class IntegratedAgent:
    """Corpo + modelo de mundo + curiosidade, num laço só.

    policy:
      • 'curious' — navega para o lugar de maior learning progress (active inference).
      • 'random'  — passeia aleatoriamente pelo anel (baseline corporificado).
    """

    def __init__(self, world: RingWorld, policy: str = "curious",
                 n_latent: int = 12, eps: float = 0.1, dwell: int = 8, seed: int = 0,
                 replay_capacity: int = 0, replay_every: int = 0, replay_batch: int = 0):
        self.world = world
        self.policy = policy
        self.eps = eps
        self.dwell_max = dwell
        self.pc = PredictiveCoder(n_obs=world.D, n_latent=n_latent, n_infer=30,
                                  eta_r=0.1, eta_w=0.05, l2_prior=0.1, seed=seed)
        self.mot = IntrinsicMotivation(n_regions=world.n)
        self.pos = 0
        self.target = None
        self._dwell = 0
        # Memória opcional (M8): replay para consolidar e não esquecer.
        self.replay_every = replay_every
        self.replay_batch = replay_batch
        self.buffer = (ReplayBuffer(replay_capacity, world.D, seed=seed)
                       if replay_capacity > 0 else None)
        self._t = 0

    def step(self, rng):
        """Um instante de vida: sente, aprende, decide e move. Retorna a posição
        ANTES de mover (onde sentiu) e o erro de previsão ali."""
        here = self.pos
        x = self.world.sense(here, rng)
        _, err = self.pc.learn(x)          # modelo de mundo aprende (local, sem backprop)
        if self.buffer is not None:        # M8: guarda e reensaia ("sono")
            self.buffer.add(x)
            if self.replay_every and self._t % self.replay_every == 0 and self.buffer.size:
                for xr in self.buffer.sample(self.replay_batch):
                    self.pc.learn(xr)      # reensaio: não move, não conta como curiosidade
        self.mot.update(here, err)         # curiosidade observa o progresso aqui
        self._navigate(rng)
        self._t += 1
        return here, err

    def _navigate(self, rng):
        n = self.world.n
        if self.policy == "random":
            self.pos = (self.pos + int(rng.integers(-1, 2))) % n
            return
        # 'curious': escolhe um alvo por learning progress e caminha até ele.
        if self.target is None or (self.pos == self.target and self._dwell >= self.dwell_max):
            if rng.random() < self.eps:
                self.target = int(rng.integers(n))           # exploração
            else:
                self.target = int(np.argmax(self.mot.learning_progress()))
            self._dwell = 0
        if self.pos == self.target:
            self._dwell += 1                                 # fica e amostra
        else:
            self.pos = self._step_toward(self.target, n)

    def _step_toward(self, target, n):
        """Um passo na direção mais curta pelo anel."""
        forward = (target - self.pos) % n
        return (self.pos + (1 if forward <= n // 2 else -1)) % n

    def mastery(self):
        """Domínio (%) por lugar aprendível: 1 - erro_atual/erro_inicial não é
        guardado aqui; retorna o erro de previsão limpo por lugar."""
        return {l: float(self.pc.prediction_error(self.world.patterns[l]))
                for l in self.world.learnable_locs()}
