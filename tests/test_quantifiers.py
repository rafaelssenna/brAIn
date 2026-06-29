"""Testes do M36 — quantificadores (todos/algum/nenhum).

Verifica que o agente avalia frases quantificadas contra a cena (semântica de verdade),
aplicando a lógica de ∀/∃/¬∃, e que confundir os quantificadores erra todos/nenhum.

Roda com:  python -m pytest tests/test_quantifiers.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

np.seterr(all="ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import QuantifierGame, ALL, SOME, NONE  # noqa: E402

KC = 3


def _trained(logic, seed=1, n=5000):
    g = QuantifierGame(KC, correct_logic=logic, seed=seed)
    g.train_grounding(n, rng_seed=seed + 50)
    return g


def test_avalia_quantificadores_corretamente():
    """Com a lógica correta, acerta V/F para todos os quantificadores."""
    g = _trained(True)
    accs = g.eval_accuracy(rng_seed=0)
    assert all(v > 0.9 for v in accs.values())


def test_logica_de_todos():
    """'todos c' = ∀: verdadeiro só quando todo objeto tem a cor c."""
    g = _trained(True)
    assert g.evaluate(ALL, 0, [0, 0, 0]) is True
    assert g.evaluate(ALL, 0, [0, 1, 0]) is False


def test_logica_de_algum_e_nenhum():
    """'algum c' = ∃; 'nenhum c' = ¬∃."""
    g = _trained(True)
    assert g.evaluate(SOME, 0, [1, 0, 2]) is True
    assert g.evaluate(SOME, 0, [1, 2, 1]) is False
    assert g.evaluate(NONE, 0, [1, 2, 1]) is True
    assert g.evaluate(NONE, 0, [1, 0, 2]) is False


def test_confundir_quantificadores_erra_todos_e_nenhum():
    """Ablação: tratar tudo como 'algum' acerta 'algum' mas erra 'todos' e 'nenhum'."""
    g = _trained(False)
    accs = g.eval_accuracy(rng_seed=0)
    assert accs["algum"] > 0.9
    assert accs["todos"] < 0.7
    assert accs["nenhum"] < 0.5


def test_logica_correta_supera_confusa():
    """Aplicar a lógica certa é claramente melhor que confundir os quantificadores."""
    assert _trained(True).overall_accuracy(rng_seed=0) > _trained(False).overall_accuracy(rng_seed=0) + 0.3
