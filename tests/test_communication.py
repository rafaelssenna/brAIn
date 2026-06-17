"""Testes do M16 — comunicação emergente (jogo de referência).

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import SignalingGame  # noqa: E402


def _train(game, rounds, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        game.play(int(rng.integers(game.nc)))


def test_comunicacao_emerge():
    """Do acaso a ~100%: os agentes convergem para um protocolo compartilhado."""
    game = SignalingGame(5, 5, lr=0.15, temp=0.4, seed=0)
    assert game.eval_accuracy() < 0.6        # antes de treinar: ~acaso
    _train(game, 5000)
    assert game.eval_accuracy() == 1.0       # depois: comunicação perfeita


def test_codigo_e_bijecao():
    """O léxico emergente é unívoco (cada conceito -> um símbolo distinto)."""
    game = SignalingGame(5, 5, lr=0.15, temp=0.4, seed=1)
    _train(game, 5000)
    assert game.is_bijection()
    assert len(set(game.lexicon())) == 5


def test_convergencia_robusta_entre_seeds():
    """Converge para comunicação perfeita em várias sementes (não é sorte)."""
    accs = []
    for s in range(5):
        g = SignalingGame(5, 5, lr=0.15, temp=0.4, seed=s)
        _train(g, 5000, seed=s)
        accs.append(g.eval_accuracy())
    assert np.mean(accs) > 0.95
