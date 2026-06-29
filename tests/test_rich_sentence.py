"""Testes do M32 — frases mais ricas (descrição de 3 atributos).

Verifica que o agente monta frases de 3 atributos só ouvindo (grounding + ordem),
GENERALIZA para cenas nunca vistas (produtividade), aprende ordens de idiomas diferentes,
e NÃO aprende quando o idioma não tem ordem fixa.

Roda com:  python -m pytest tests/test_rich_sentence.py -v
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import RichSentenceLearner, make_rich_language  # noqa: E402

K = [3, 3, 3]
ALL = [(s, c, p) for s in range(3) for c in range(3) for p in range(3)]
ORDER_PT = (0, 1, 2)
ORDER_X = (2, 0, 1)


def _teach(order, train, n=4000, seed=1):
    lang = make_rich_language(order, K)
    ag = RichSentenceLearner(K, seed=seed)
    rng = np.random.default_rng(0)
    for _ in range(n):
        v = train[int(rng.integers(len(train)))]
        ag.hear(lang(v), v)
    return ag, lang


def test_monta_frase_de_3_atributos():
    """Ouvindo o idioma, aprende a ordem e produz/entende a frase de 3 atributos."""
    ag, lang = _teach(ORDER_PT, ALL)
    assert ag.learned_order() == ORDER_PT
    assert np.mean([ag.say(v) == lang(v) for v in ALL]) == 1.0
    assert np.mean([ag.understand(lang(v)) == v for v in ALL]) == 1.0


def test_generaliza_para_cenas_nunca_vistas():
    """Treina em 18 de 27 cenas e monta frases certas para as 9 NUNCA vistas (produtividade)."""
    rng = np.random.default_rng(3)
    idx = rng.permutation(len(ALL))
    train = [ALL[i] for i in idx[:18]]; test = [ALL[i] for i in idx[18:]]
    ag, lang = _teach(ORDER_PT, train)
    assert np.mean([ag.say(v) == lang(v) for v in test]) > 0.9


def test_mesmo_mecanismo_aprende_outra_ordem():
    """O mesmo código aprende a ordem de outro idioma (modificadores trocados)."""
    ag, lang = _teach(ORDER_X, ALL)
    assert ag.learned_order() == ORDER_X
    assert np.mean([ag.say(v) == lang(v) for v in ALL]) == 1.0


def test_papeis_dos_slots_sao_distintos():
    """A atribuição de papéis é uma divisão de trabalho: os 3 slots carregam os 3 atributos
    distintos (nenhum atributo em dois slots)."""
    ag, _ = _teach(ORDER_PT, ALL)
    roles = ag.slot_roles()
    assert sorted(roles) == [0, 1, 2]


def test_sem_ordem_fixa_nao_aprende():
    """Controle: se o idioma embaralha a ordem a cada frase, entender despenca."""
    perms = list(itertools.permutations(range(3)))
    rr = np.random.default_rng(99)
    off = [0, 3, 6]

    def lang_rand(v):
        o = perms[int(rr.integers(len(perms)))]
        return [off[a] + v[a] for a in o]

    ag = RichSentenceLearner(K, seed=1)
    r = np.random.default_rng(0)
    for _ in range(4000):
        v = ALL[int(r.integers(len(ALL)))]
        ag.hear(lang_rand(v), v)
    und = np.mean([ag.understand(lang_rand(v)) == v for v in ALL for _ in range(3)])
    assert und < 0.6                                    # bem abaixo de 100% (sem estrutura fixa)
