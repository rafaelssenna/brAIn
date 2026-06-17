"""O organismo integrado — M13 (a costura final da Fase 2).

Junta num ÚNICO agente, num só laço, tudo o que foi construído isolado:
  • CORPO num grid com paredes (M3/M7)
  • MODELO DE TRANSIÇÃO aprendido → PLANEJAMENTO de rotas (M11)
  • PREDICTIVE CODER HIERÁRQUICO = modelo de mundo / conceitos (M4/M9)
  • CURIOSIDADE (learning progress) escolhendo as METAS (M5)
  • MEMÓRIA / replay para não esquecer (M8)

A divisão de trabalho que emerge: a curiosidade decide PARA ONDE ir; o
planejamento descobre COMO chegar (contornando paredes que o reativo não vence);
a hierarquia APRENDE o que há lá; a memória RETÉM. É a H1 costurada — em
miniatura, rate-based (o spiking do M10 fica como substrato futuro), honesta.
"""

from __future__ import annotations

import numpy as np

from .hierarchy import HierarchicalPredictiveCoder
from .curiosity import IntrinsicMotivation
from .memory import ReplayBuffer
from .planning import WorldModel, plan


class IntegratedBrain:
    """O cérebro completo num laço só. `grid` é um GridWorld (transições/paredes);
    `patterns` mapeia célula→padrão aprendível; `noise_cells` são lugares de ruído."""

    def __init__(self, grid, patterns, noise_cells, start, sizes,
                 use_planning: bool = True, use_memory: bool = True,
                 replay_every: int = 15, replay_batch: int = 8, seed: int = 0):
        self.grid = grid
        self.patterns = patterns
        self.noise_cells = set(noise_cells)
        self.content_cells = list(patterns.keys()) + list(self.noise_cells)
        self.idx = {c: i for i, c in enumerate(self.content_cells)}

        self.hpc = HierarchicalPredictiveCoder(sizes, seed=seed)     # M9
        self.tm = WorldModel()                                       # M11
        self.mot = IntrinsicMotivation(len(self.content_cells))      # M5
        self.buffer = ReplayBuffer(400, sizes[0], seed=seed) if use_memory else None  # M8

        self.use_planning = use_planning
        self.replay_every = replay_every
        self.replay_batch = replay_batch
        self.D = sizes[0]
        self.blank = np.zeros(self.D)
        self.pos = start
        self.goal = None
        self._t = 0

    # --- percepção ---
    def _obs(self, cell, rng):
        if cell in self.noise_cells:
            v = rng.standard_normal(self.D)
            return v / np.linalg.norm(v)
        if cell in self.patterns:
            v = self.patterns[cell] + 0.05 * rng.standard_normal(self.D)
            return v / np.linalg.norm(v)
        return self.blank                                  # corredor: nada a aprender

    def err_on(self, pattern):
        return float(np.sum((pattern - self.hpc.reconstruct(pattern)) ** 2))

    def mastery(self):
        return {c: self.err_on(self.patterns[c]) for c in self.patterns}

    # --- um instante de vida ---
    def step(self, rng):
        here = self.pos
        x = self._obs(here, rng)
        _, err = self.hpc.learn(x)                          # aprende o mundo (hierárquico)
        if here in self.idx:
            self.mot.update(self.idx[here], err)            # curiosidade observa o progresso
        if self.buffer is not None:                         # memória: guarda e reensaia
            self.buffer.add(x)
            if self._t % self.replay_every == 0 and self.buffer.size:
                for xr in self.buffer.sample(self.replay_batch):
                    self.hpc.learn(xr)
        self._navigate(rng)
        self._t += 1
        return here, err

    # --- decisão: curiosidade escolhe a meta, planejamento traça a rota ---
    def _choose_goal(self, rng):
        lp = self.mot.learning_progress()
        if rng.random() < 0.15:
            gi = int(rng.integers(len(self.content_cells)))
        else:
            gi = int(np.argmax(lp))
        self.goal = self.content_cells[gi]

    def _navigate(self, rng):
        if self.goal is None or self.pos == self.goal:
            self._choose_goal(rng)
        if self.use_planning:
            path = plan(self.tm, self.pos, self.goal)       # PENSA a rota (M11)
            a = path[0] if path else int(rng.integers(4))   # sem rota (mapa incompleto) => explora
        else:
            dr, dc = self.goal[0] - self.pos[0], self.goal[1] - self.pos[1]
            a = (1 if dr > 0 else 0) if abs(dr) >= abs(dc) else (3 if dc > 0 else 2)
        sp = self.grid.step(self.pos, a)
        self.tm.observe(self.pos, a, sp)                    # aprende a transição vivendo
        if sp == self.pos:                                  # travou: cutucada aleatória (e aprende)
            a2 = int(rng.integers(4))
            sp = self.grid.step(self.pos, a2)
            self.tm.observe(self.pos, a2, sp)
        self.pos = sp
