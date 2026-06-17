"""Testes do M17 — proto-sintaxe / composicionalidade.

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import CompositionalGame  # noqa: E402

K1 = K2 = 3
ALL = [(a, b) for a in range(K1) for b in range(K2)]
HELDOUT = [(0, 0), (1, 1), (2, 2)]
TRAIN = [c for c in ALL if c not in HELDOUT]


def _train(mode, seed=0, rounds=6000):
    g = CompositionalGame(K1, K2, 4, mode=mode, lr=0.15, temp=0.4, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        a1, a2 = TRAIN[rng.integers(len(TRAIN))]
        g.play(a1, a2)
    return g


def test_composicional_generaliza_para_nunca_visto():
    """O código composicional acerta combinações NUNCA vistas no treino."""
    g = _train("compositional")
    assert g.accuracy(TRAIN) > 0.95
    assert g.accuracy(HELDOUT) > 0.95        # generaliza (produtividade)


def test_holistico_nao_generaliza():
    """O código holístico memoriza o treino mas falha no nunca-visto."""
    g = _train("holistic")
    assert g.accuracy(TRAIN) > 0.95
    assert g.accuracy(HELDOUT) < 0.5         # não generaliza


def test_generalizacao_composicional_robusta():
    """Composicional generaliza em várias sementes (não é sorte)."""
    ho = []
    for s in range(5):
        g = _train("compositional", seed=s)
        ho.append(g.accuracy(HELDOUT))
    assert np.mean(ho) > 0.9
