"""Testes do M27 — a primeira frase ancorada, aprendida ouvindo.

Verifica que o agente aprende a FRASE (forma × posição) só ouvindo: descobre o grounding
(palavra↔atributo) E a ordem do idioma; produz e entende em PT e EN (ordens diferentes);
GENERALIZA para combinações nunca vistas; e NÃO aprende quando o idioma não tem ordem fixa.

Roda com:  python -m pytest tests/test_grounded_sentence.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import GroundedSentenceLearner, make_language, PT_ORDER, EN_ORDER  # noqa: E402

K_SHAPE = 3
K_POS = 3
ALL = [(s, p) for s in range(K_SHAPE) for p in range(K_POS)]


def _teach(order, combos, n_steps=2000, seed=1):
    lang = make_language(order, K_SHAPE)
    ag = GroundedSentenceLearner(K_SHAPE, K_POS, seed=seed)
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        s, p = combos[int(rng.integers(len(combos)))]
        ag.hear(lang(s, p), s, p)
    return ag, lang


def test_aprende_a_ordem_e_a_frase_em_pt():
    """Ouvindo português ([forma][posição]), aprende a ordem e produz/entende 100%."""
    ag, lang = _teach(PT_ORDER, ALL)
    assert ag.learned_order() == PT_ORDER
    prod = np.mean([ag.say(s, p) == lang(s, p) for (s, p) in ALL])
    und = np.mean([ag.understand(lang(s, p)) == (s, p) for (s, p) in ALL])
    assert prod == 1.0 and und == 1.0


def test_mesmo_mecanismo_aprende_ingles():
    """O MESMO código, ouvindo inglês ([posição][forma]), aprende a OUTRA ordem e acerta 100%."""
    ag, lang = _teach(EN_ORDER, ALL)
    assert ag.learned_order() == EN_ORDER             # ordem diferente do PT
    prod = np.mean([ag.say(s, p) == lang(s, p) for (s, p) in ALL])
    und = np.mean([ag.understand(lang(s, p)) == (s, p) for (s, p) in ALL])
    assert prod == 1.0 and und == 1.0


def test_ordens_aprendidas_sao_diferentes_entre_idiomas():
    """PT e EN produzem a frase em ORDENS distintas para a mesma cena."""
    ag_pt, lang_pt = _teach(PT_ORDER, ALL)
    ag_en, lang_en = _teach(EN_ORDER, ALL)
    assert ag_pt.say(0, 0) != ag_en.say(0, 0)         # "barraH topo" != "topo barraH"
    assert ag_pt.learned_order() != ag_en.learned_order()


def test_generaliza_para_combinacoes_nunca_vistas():
    """Treina em 6 de 9 combinações e monta frases certas para as 3 NUNCA vistas (produtividade)."""
    rng = np.random.default_rng(3)
    idx = rng.permutation(len(ALL))
    train = [ALL[i] for i in idx[:6]]
    test = [ALL[i] for i in idx[6:]]
    ag, lang = _teach(PT_ORDER, train)
    gen = np.mean([ag.say(s, p) == lang(s, p) for (s, p) in test])
    assert gen == 1.0                                  # generaliza para o nunca-visto


def test_sem_ordem_fixa_nao_aprende_a_sintaxe():
    """Controle: se o idioma embaralha a ordem a cada frase, entender despenca."""
    rr = np.random.default_rng(99)

    def lang_rand(s, p):
        order = [PT_ORDER, EN_ORDER][int(rr.integers(2))]
        w = {"shape": s, "pos": K_SHAPE + p}
        return [w[role] for role in order]

    ag = GroundedSentenceLearner(K_SHAPE, K_POS, seed=1)
    r = np.random.default_rng(0)
    for _ in range(2000):
        s = int(r.integers(K_SHAPE)); p = int(r.integers(K_POS))
        ag.hear(lang_rand(s, p), s, p)
    und = np.mean([ag.understand(lang_rand(s, p)) == (s, p)
                   for s in range(K_SHAPE) for p in range(K_POS) for _ in range(5)])
    # ordem fixa dá 100%; sem ordem fixa fica bem abaixo (controle de estrutura real)
    assert und < 0.85
