"""Testes do M13 — o organismo integrado (a costura final).

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import GridWorld, IntegratedBrain  # noqa: E402

SIDE = 4
D = SIDE * SIDE
C_CELL = (0, 4)


def _bar(kind, i):
    p = np.zeros((SIDE, SIDE))
    if kind == "h":
        p[i, :] = 1.0
    else:
        p[:, i] = 1.0
    return p.ravel() / np.linalg.norm(p)


def _setup(use_planning):
    grid = GridWorld(5, [(r, 2) for r in range(4)], goal=None)   # parede, passagem em (4,2)
    patterns = {(0, 0): _bar("h", 0), (3, 1): _bar("h", 2), C_CELL: _bar("v", 0)}
    return IntegratedBrain(grid, patterns, [(4, 4)], (0, 0), [D, 12, 6],
                           use_planning=use_planning, use_memory=True, seed=0)


def test_so_planejando_alcanca_C_atras_da_parede():
    """O resultado robusto do M13: o organismo que planeja alcança C (atrás da
    parede); o reativo NUNCA chega lá."""
    rng = np.random.default_rng(0)
    brain = _setup(use_planning=True)
    visits_plan = sum(brain.step(rng)[0] == C_CELL for _ in range(2500))

    rng = np.random.default_rng(0)
    react = _setup(use_planning=False)
    visits_react = sum(react.step(rng)[0] == C_CELL for _ in range(2500))

    assert visits_plan > 0          # planejando, chega em C
    assert visits_react == 0        # reagindo, nunca contorna a parede


def test_organismo_aprende_o_mundo():
    """O modelo de mundo melhora: o erro de previsão num padrão cai com a vida."""
    rng = np.random.default_rng(0)
    brain = _setup(use_planning=True)
    p = brain.patterns[(0, 0)]
    e0 = brain.err_on(p)
    for _ in range(2000):
        brain.step(rng)
    assert brain.err_on(p) < e0     # aprendeu (modelo de mundo melhorou)
