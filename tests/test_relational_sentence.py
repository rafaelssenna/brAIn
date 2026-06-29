"""Testes do M33 — frases relacionais e recursão.

Verifica que o agente compõe e entende frases recursivas (objeto-relação-objeto) e
generaliza para cenas relacionais nunca vistas.

Roda com:  python -m pytest tests/test_relational_sentence.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import RelationalSentenceLearner, make_relational_language  # noqa: E402

KS, KC, KR = 3, 3, 2
SCENES = [((s1, c1), r, (s2, c2)) for s1 in range(KS) for c1 in range(KC)
          for r in range(KR) for s2 in range(KS) for c2 in range(KC)]


def _teach(order, train, n=6000, seed=1):
    lang = make_relational_language(order, KS, KC)
    ag = RelationalSentenceLearner(KS, KC, KR, seed=seed)
    rng = np.random.default_rng(0)
    for _ in range(n):
        o1, r, o2 = train[int(rng.integers(len(train)))]
        ag.hear(lang(o1, r, o2), o1, r, o2)
    return ag, lang


def test_compoe_e_entende_frase_relacional():
    """Aprende a frase relacional inteira (objeto-relação-objeto)."""
    ag, lang = _teach((0, 1), SCENES)
    assert np.mean([ag.say(*c) == lang(*c) for c in SCENES]) > 0.95
    assert np.mean([ag.understand(lang(*c)) == c for c in SCENES]) > 0.95


def test_generaliza_para_relacionais_nunca_vistas():
    """Treina em 70% e compõe frases relacionais certas para as nunca vistas (recursão)."""
    rng = np.random.default_rng(3)
    idx = rng.permutation(len(SCENES))
    ntr = int(0.7 * len(SCENES))
    train = [SCENES[i] for i in idx[:ntr]]; test = [SCENES[i] for i in idx[ntr:]]
    ag, lang = _teach((0, 1), train)
    assert np.mean([ag.say(*c) == lang(*c) for c in test]) > 0.9


def test_estrutura_recursiva_da_frase():
    """A frase tem a estrutura sub(obj1) + relação + sub(obj2): tamanho = 2 + 1 + 2 = 5."""
    ag, lang = _teach((0, 1), SCENES)
    sent = ag.say((0, 0), 0, (1, 2))
    assert len(sent) == 5
    # a palavra do meio é de relação (>= bloco de objeto)
    assert sent[2] >= (KS + KC)


def test_relacao_e_objetos_decodificam_separados():
    """Entender a frase devolve os dois objetos e a relação corretamente."""
    ag, lang = _teach((0, 1), SCENES)
    o1, r, o2 = (2, 1), 1, (0, 2)
    dec1, dr, dec2 = ag.understand(lang(o1, r, o2))
    assert dec1 == o1 and dr == r and dec2 == o2


def test_sub_frase_de_objeto_e_reusada():
    """A mesma sub-frase descreve o objeto, esteja ele na 1a ou na 2a posição (recursão)."""
    ag, lang = _teach((0, 1), SCENES)
    obj = (1, 2)
    # objeto na posição 1 vs na posição 2 -> a sub-frase do objeto é a mesma
    s_left = ag.say(obj, 0, (0, 0))[:2]
    s_right = ag.say((0, 0), 0, obj)[3:]
    assert s_left == s_right
