"""Agência / planejamento — M11 (Fase 2): o primeiro "pensar".

"Pensar = rodar a previsão por dentro." O agente aprende um MODELO DE MUNDO
(o que cada ação faz: (estado, ação) -> próximo estado) vivendo — nasce sem saber.
Depois, para decidir, ele SIMULA ações na imaginação (busca no modelo) e escolhe
o caminho até o objetivo — sem precisar tentar no mundo real. Isso é planejar.

Contraste: um agente REATIVO, sem modelo, só anda na direção do objetivo e fica
preso quando há uma parede no caminho (não imagina o desvio).

Conexão com a visão: é a MESMA máquina de previsão dos marcos anteriores, agora
usada para AÇÃO (imaginar consequências) e não só percepção — a ponte para a
cognição emergente da Fase 3.
"""

from __future__ import annotations

from collections import deque

import numpy as np

# Ações: 0=cima, 1=baixo, 2=esquerda, 3=direita.
DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = 4


class GridWorld:
    """Labirinto quadrado com paredes. Mover para parede/fora => fica no lugar."""

    def __init__(self, size: int, walls, goal):
        self.size = size
        self.walls = set(walls)
        self.goal = goal

    def in_bounds(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size

    def free_states(self):
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if (r, c) not in self.walls]

    def step(self, s, a):
        dr, dc = DELTAS[a]
        nxt = (s[0] + dr, s[1] + dc)
        if not self.in_bounds(nxt) or nxt in self.walls:
            return s
        return nxt


class WorldModel:
    """Modelo de transição APRENDIDO pela experiência. O agente nasce sem saber
    o que suas ações fazem; preenche (s,a)->s' ao viver."""

    def __init__(self):
        self.T = {}

    def observe(self, s, a, sp):
        self.T[(s, a)] = sp

    def predict(self, s, a):
        return self.T.get((s, a), s)          # desconhecido => supõe que fica parado

    def known(self, s, a):
        return (s, a) in self.T

    def coverage(self, world):
        total = len(world.free_states()) * N_ACTIONS
        return len(self.T) / total if total else 0.0


def explore_and_learn(world: GridWorld, model: WorldModel, n_steps: int, rng,
                      restart_every: int = 30):
    """Random walk com reinícios; o agente observa cada transição e aprende."""
    states = world.free_states()
    s = states[rng.integers(len(states))]
    for t in range(n_steps):
        if t % restart_every == 0:
            s = states[rng.integers(len(states))]      # reinício episódico
        a = int(rng.integers(N_ACTIONS))
        sp = world.step(s, a)
        model.observe(s, a, sp)                        # aprende vivendo
        s = sp


def plan(model: WorldModel, start, goal):
    """PENSA: busca em largura na IMAGINAÇÃO (usando o modelo) o caminho até o
    objetivo. Retorna a sequência de ações, ou None se não achar."""
    came = {start: None}
    q = deque([start])
    while q:
        s = q.popleft()
        if s == goal:
            break
        for a in range(N_ACTIONS):
            sp = model.predict(s, a)
            if sp not in came:
                came[sp] = (s, a)
                q.append(sp)
    if goal not in came:
        return None
    actions, s = [], goal
    while came[s] is not None:
        ps, a = came[s]
        actions.append(a)
        s = ps
    return actions[::-1]


def reactive_action(s, goal, rng, stuck):
    """Política reativa SEM modelo: anda na direção do objetivo; se travar, chuta
    uma direção aleatória (não imagina o desvio)."""
    if stuck:
        return int(rng.integers(N_ACTIONS))
    dr, dc = goal[0] - s[0], goal[1] - s[1]
    if abs(dr) >= abs(dc):
        return 1 if dr > 0 else 0
    return 3 if dc > 0 else 2


def run_planner(world, model, start, goal, max_steps=200):
    """Executa o plano imaginado no mundo real. Retorna (chegou, nº de passos)."""
    actions = plan(model, start, goal)
    if not actions:
        return False, max_steps
    s = start
    for i, a in enumerate(actions):
        s = world.step(s, a)
        if s == goal:
            return True, i + 1
    return s == goal, len(actions)


def run_reactive(world, start, goal, rng, max_steps=200):
    """Agente reativo (sem imaginação). Retorna (chegou, nº de passos)."""
    s = start
    for t in range(max_steps):
        if s == goal:
            return True, t
        a = reactive_action(s, goal, rng, stuck=False)
        sp = world.step(s, a)
        if sp == s:                                    # travou: tenta aleatório
            sp = world.step(s, reactive_action(s, goal, rng, stuck=True))
        s = sp
    return s == goal, max_steps
