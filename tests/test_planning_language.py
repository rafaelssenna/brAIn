"""Testes do M26 — o organismo que planeja rotas num mundo 2D e fala.

Verifica que o organismo COMPLETO num grid com PAREDE: aprende o modelo de transição,
planeja rotas, alcança o objeto ATRÁS da parede (que o reativo nunca alcança), aprende
a percebê-lo e a NOMEÁ-LO (atenção conjunta, M21) — algo que o reativo nunca consegue.

Roda com:  python -m pytest tests/test_planning_language.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

np.seterr(all="ignore")     # warnings espúrios de matmul (NumPy/BLAS); resultados são finitos

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import GridWorld  # noqa: E402
from brain.planning_language import PlanningLanguageAgent  # noqa: E402

GRID = 5
SIDE = 4
D = SIDE * SIDE
START = (0, 0)
C_CELL = (0, 4)                                  # atrás da parede
WALL = [(r, 2) for r in range(4)]               # passagem em (4,2)


def _bar(kind, i):
    p = np.zeros((SIDE, SIDE))
    if kind == "h":
        p[i, :] = 1.0
    else:
        p[:, i] = 1.0
    v = p.ravel()
    return v / np.linalg.norm(v)


PATTERNS = {(0, 0): _bar("h", 0), (3, 1): _bar("h", 2), C_CELL: _bar("v", 0)}
CONTENT_CELLS = list(PATTERNS.keys())
CLEAN = [PATTERNS[c] for c in CONTENT_CELLS]
C_IDX = CONTENT_CELLS.index(C_CELL)


def _grid():
    return GridWorld(GRID, WALL, goal=None)


def _obs_for(cell, rng):
    if cell in PATTERNS:
        v = PATTERNS[cell] + 0.05 * rng.standard_normal(D)
        return v / np.linalg.norm(v)
    return np.zeros(D)


def _naming(A):
    return [int(A.speak(A.concept_of(CLEAN[o]), explore=False) == o) for o in range(len(CLEAN))]


def _live(use_planning, n_steps, seed_rng=42):
    A = PlanningLanguageAgent(_grid(), CONTENT_CELLS, D, 12, len(CONTENT_CELLS),
                              START, use_planning=use_planning, eps=0.15, seed=1)
    rng = np.random.default_rng(seed_rng)
    visits_C = 0
    for t in range(n_steps):
        here, _, x = A.perceive_and_learn(_obs_for, rng)
        A.learn_word_here(x)                          # professor nomeia (atenção conjunta)
        visits_C += int(here == C_CELL)
        A.navigate()
    return A, visits_C


def test_planejador_alcanca_objeto_atras_da_parede():
    """O planejador imagina a rota e CHEGA ao objeto atrás da parede; o reativo não."""
    _, vis_plan = _live(True, n_steps=3000)
    _, vis_react = _live(False, n_steps=3000)
    assert vis_plan > 0                          # planejador chega a C
    assert vis_react == 0                         # reativo trava na parede, nunca chega a C


def test_so_quem_planeja_aprende_a_ver_C():
    """Só quem alcança C aprende a percebê-lo: surpresa em C cai no planejador, não no reativo."""
    A_plan, _ = _live(True, n_steps=3500)
    A_react, _ = _live(False, n_steps=3500)
    assert A_plan.surprise_on(PATTERNS[C_CELL]) < 0.1
    assert A_react.surprise_on(PATTERNS[C_CELL]) > 0.3


def test_so_quem_planeja_aprende_a_nomear_C():
    """O coração do M26: só quem planeja a rota aprende a NOMEAR o objeto atrás da parede."""
    A_plan, _ = _live(True, n_steps=4000)
    A_react, _ = _live(False, n_steps=4000)
    nm_plan = _naming(A_plan)
    nm_react = _naming(A_react)
    assert nm_plan[C_IDX] == 1                    # planejador nomeia C
    assert nm_react[C_IDX] == 0                    # reativo NUNCA nomeia C (nunca chegou)
    # ambos nomeiam os objetos acessíveis (a diferença é só o que está atrás da parede)
    assert sum(nm_plan) > sum(nm_react)


def test_aprende_o_modelo_de_transicao_vivendo():
    """O agente nasce sem saber o que suas ações fazem e aprende as transições vivendo."""
    A, _ = _live(True, n_steps=2000)
    assert len(A.tm.T) > 0
    assert any(s == START for (s, a) in A.tm.T)


def test_planejador_compoe_com_spiking():
    """O organismo 2D que planeja roda também com percepção spiking (M24) — compõe."""
    A = PlanningLanguageAgent(_grid(), CONTENT_CELLS, D, 12, len(CONTENT_CELLS),
                              START, use_planning=True, spiking=True, seed=1)
    from brain.spiking_predictive import SpikingPerceptionCoder
    assert isinstance(A.mind.pc, SpikingPerceptionCoder)
    rng = np.random.default_rng(0)
    for _ in range(300):
        _, _, x = A.perceive_and_learn(_obs_for, rng)
        A.learn_word_here(x); A.navigate()
    assert len(A.tm.T) > 0
