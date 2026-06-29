"""Testes do M31 — correferência: resolver 'dela' entre vários objetos.

Verifica que, com vários candidatos, o agente resolve a referência escolhendo o objeto
certo — pela RECÊNCIA (o mais recente) ou por DESCRIÇÃO (o atributo dito) — algo que
escolher ao acaso entre os candidatos não consegue.

Roda com:  python -m pytest tests/test_coreference.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

np.seterr(all="ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import CoreferenceDialogue  # noqa: E402

K_SHAPE = 3
K_POS = 3
ALL = [(s, p) for s in range(K_SHAPE) for p in range(K_POS)]
PAIRS = [(o1, o2) for o1 in ALL for o2 in ALL if o1 != o2][:40]


def _trained(strategy, seed=1, n=8000):
    g = CoreferenceDialogue(K_SHAPE, K_POS, strategy=strategy, seed=seed)
    g.train_grounding(ALL, n, rng_seed=seed + 50)
    return g


def test_recencia_resolve_o_referente_certo():
    """Pela recência, 'dela' resolve o objeto mencionado mais recentemente."""
    g = _trained("recency")
    assert g.eval_recency(PAIRS, ask="pos") > 0.9


def test_aleatorio_cai_ao_acaso():
    """Ablação: escolher um candidato ao acaso resolve só ~metade (2 candidatos)."""
    g = _trained("random")
    assert g.eval_recency(PAIRS, ask="pos") < 0.7        # ~50% com 2 candidatos


def test_recencia_supera_aleatorio():
    """Resolver pela recência é claramente melhor que escolher ao acaso."""
    g_rec = _trained("recency")
    g_rnd = _trained("random")
    assert g_rec.eval_recency(PAIRS, ask="pos") > g_rnd.eval_recency(PAIRS, ask="pos") + 0.25


def test_descricao_desambigua_pelo_atributo():
    """A descrição ('a barraX...') escolhe o objeto que casa com o atributo, mesmo não
    sendo o mais recente."""
    triples = [(o1, o2, ("shape", o1[0])) for o1 in ALL for o2 in ALL if o1[0] != o2[0]][:40]
    g = _trained("recency")
    assert g.eval_described(triples, ask="pos") > 0.9


def test_pilha_de_foco_prioriza_o_recente():
    """A pilha de foco coloca o objeto mencionado por último no topo (mais saliente)."""
    g = _trained("recency")
    o1, o2 = (0, 0), (1, 2)
    g.reset(); g.mention(o1); g.mention(o2)
    assert g.resolve() == o2                              # o mais recente
    # rementionar o1 traz ele de volta ao topo
    g.mention(o1)
    assert g.resolve() == o1
