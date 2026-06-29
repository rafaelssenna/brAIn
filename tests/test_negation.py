"""Testes do M34 — negação e contraste.

Verifica que o agente entende a NEGAÇÃO ('não X' = o complemento de X): aponta o objeto
que NÃO tem a cor negada. Um agente que ignora o 'não' escolhe exatamente o errado.

Roda com:  python -m pytest tests/test_negation.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

np.seterr(all="ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import NegationGame  # noqa: E402

KC = 3
PAIRS = [(a, b) for a in range(KC) for b in range(KC) if a != b]


def _trained(neg, seed=1, n=6000):
    g = NegationGame(KC, understand_negation=neg, seed=seed)
    g.train_grounding(n, rng_seed=seed + 50)
    return g


def test_entende_negacao_escolhe_o_complemento():
    """'não X' -> o agente aponta o objeto que NÃO tem a cor X (o complemento)."""
    g = _trained(True)
    assert g.eval_negation(PAIRS) > 0.9


def test_ignorar_negacao_escolhe_o_errado():
    """Ablação: ignorar o 'não' faz escolher o objeto NEGADO — exatamente o errado."""
    g = _trained(False)
    assert g.eval_negation(PAIRS) < 0.2               # pega o negado, quase sempre errado


def test_entender_supera_ignorar():
    """Entender a negação é drasticamente melhor que ignorá-la."""
    g_yes = _trained(True)
    g_no = _trained(False)
    assert g_yes.eval_negation(PAIRS) > g_no.eval_negation(PAIRS) + 0.5


def test_grounding_de_cor_funciona():
    """O grounding de cor (palavra<->cor) que sustenta a negação está correto."""
    g = _trained(True)
    for c in range(KC):
        assert g.decode_color(g.color_word(c)) == c   # nomeia e decodifica a mesma cor


def test_negacao_e_o_operador_complemento():
    """Para um par concreto, 'não a' resolve o objeto b (o complemento), não a."""
    g = _trained(True)
    a, b = 0, 2
    word = g.color_word(a)
    choice = g.resolve_negation(word, [a, b])
    assert [a, b][choice] == b                         # escolheu o complemento
