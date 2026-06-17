"""Testes do M11 — agência / planejamento (o primeiro "pensar").

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import (GridWorld, WorldModel, explore_and_learn, plan,  # noqa: E402
                   run_planner, run_reactive)


def _maze(goal):
    walls = [(3, c) for c in range(5)]      # parede com passagem nas colunas 5,6
    return GridWorld(7, walls, goal)


def test_parede_bloqueia_movimento():
    w = _maze((0, 0))
    assert w.step((2, 0), 1) == (2, 0)       # tentar descer para a parede => fica
    assert w.step((2, 0), 3) == (2, 1)       # mover para célula livre => anda
    assert w.step((0, 0), 0) == (0, 0)       # fora dos limites => fica


def test_modelo_aprende_transicoes():
    w = _maze((0, 0)); m = WorldModel()
    assert not m.known((5, 5), 3)
    explore_and_learn(w, m, 3000, np.random.default_rng(0))
    assert m.coverage(w) > 0.9               # aprendeu quase tudo
    # o modelo prevê corretamente uma transição conhecida
    s = (5, 5)
    assert m.predict(s, 3) == w.step(s, 3)


def test_planejador_chega_onde_reativo_trava():
    """No cenário de desvio, o planejador chega e o reativo NÃO."""
    start, goal = (6, 0), (0, 0)
    w = _maze(goal); m = WorldModel()
    explore_and_learn(w, m, 4000, np.random.default_rng(0))
    ok_p, _ = run_planner(w, m, start, goal)
    ok_r, _ = run_reactive(w, start, goal, np.random.default_rng(0), max_steps=60)
    assert ok_p and not ok_r


def test_plano_imaginado_bate_com_a_realidade():
    """Executar o plano no mundo real reproduz a trajetória imaginada no modelo."""
    start, goal = (6, 0), (0, 0)
    w = _maze(goal); m = WorldModel()
    explore_and_learn(w, m, 4000, np.random.default_rng(0))
    actions = plan(m, start, goal)
    assert actions is not None
    # imaginado
    s = start; imag = [s]
    for a in actions:
        s = m.predict(s, a); imag.append(s)
    # real
    s = start; real = [s]
    for a in actions:
        s = w.step(s, a); real.append(s)
    assert imag == real
    assert real[-1] == goal
