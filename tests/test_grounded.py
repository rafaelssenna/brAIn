"""Testes do M18 — comunicação ancorada na percepção.

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import GroundedLanguageGame  # noqa: E402

K = 6
D = 16


def _objects(seed=1):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (K, D))
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def _train(g, rounds=5000, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        g.play(int(rng.integers(K)))


def test_comunicacao_ancorada_funciona_alinhada():
    """Com percepções alinhadas (sem ruído), a comunicação ancorada vai a ~100%."""
    P = _objects()
    g = GroundedLanguageGame(P, recv_noise=0.0, lr=0.15, temp=0.4, seed=0)
    assert g.discriminability() == 1.0          # receptor distingue todos
    _train(g)
    assert g.accuracy() > 0.95


def test_percepcao_desalinhada_degrada_comunicacao():
    """Percepção do receptor degradada => discriminabilidade cai => comunicação cai."""
    P = _objects()
    aligned = GroundedLanguageGame(P, recv_noise=0.0, seed=0); _train(aligned)
    # ruído alto o suficiente para fundir objetos (discriminabilidade < 1)
    noisy = GroundedLanguageGame(P, recv_noise=1.5, seed=3); _train(noisy, seed=3)
    assert noisy.discriminability() < 1.0
    assert noisy.accuracy() < aligned.accuracy()


def test_comunicacao_limitada_pela_discriminabilidade():
    """A comunicação não ultrapassa muito o que o receptor consegue distinguir."""
    P = _objects()
    g = GroundedLanguageGame(P, recv_noise=1.2, seed=5)
    _train(g, seed=5)
    # comm não pode exceder a discriminabilidade por uma margem grande
    assert g.accuracy() <= g.discriminability() + 0.15
