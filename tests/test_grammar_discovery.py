"""Testes do M35 — descobrir a gramática por estatística pura (sem rótulos).

Verifica que o agente, recebendo SÓ um corpus de frases (sem rótulos), descobre as classes
de palavras e a gramática, e distingue frases gramaticais de embaralhadas.

Roda com:  python -m pytest tests/test_grammar_discovery.py -v
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

np.seterr(all="ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import GrammarDiscovery  # noqa: E402

SHAPES, COLORS, POS = [0, 1, 2], [3, 4, 5], [6, 7, 8]
TRUE = {**{w: 0 for w in SHAPES}, **{w: 1 for w in COLORS}, **{w: 2 for w in POS}}


def _trained(seed=1, n=3000):
    rng = np.random.default_rng(seed)
    corpus = [[rng.choice(SHAPES), rng.choice(COLORS), rng.choice(POS)] for _ in range(n)]
    gd = GrammarDiscovery(9, 3, seed=seed)
    gd.observe_corpus(corpus)
    gd.discover_grammar()
    return gd


def test_descobre_classes_sem_rotulos():
    """Sem rótulos, agrupa as palavras nas classes certas (formas/cores/posições)."""
    gd = _trained()
    assert gd.class_purity(TRUE) > 0.9


def test_palavras_da_mesma_classe_ficam_juntas():
    """Todas as formas caem numa classe só; idem cores; idem posições."""
    gd = _trained()
    assert len(set(gd.labels[w] for w in SHAPES)) == 1
    assert len(set(gd.labels[w] for w in COLORS)) == 1
    assert len(set(gd.labels[w] for w in POS)) == 1
    # e as três classes são distintas entre si
    assert len({gd.labels[SHAPES[0]], gd.labels[COLORS[0]], gd.labels[POS[0]]}) == 3


def test_reconhece_frase_gramatical():
    """Uma frase na ordem do corpus (forma-cor-posição) é julgada gramatical."""
    gd = _trained()
    rng = np.random.default_rng(0)
    ok = np.mean([gd.is_grammatical([rng.choice(SHAPES), rng.choice(COLORS), rng.choice(POS)])
                  for _ in range(50)])
    assert ok > 0.9


def test_rejeita_frase_embaralhada():
    """Uma frase com a ordem trocada é julgada agramatical."""
    gd = _trained()
    rng = np.random.default_rng(0)
    perms = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]
    def bad():
        base = [rng.choice(SHAPES), rng.choice(COLORS), rng.choice(POS)]
        p = perms[rng.integers(len(perms))]
        return [base[i] for i in p]
    rej = np.mean([not gd.is_grammatical(bad()) for _ in range(50)])
    assert rej > 0.8


def test_aprende_so_da_estatistica():
    """O método é só contagem de co-ocorrência: a matriz de transição entre classes não é
    uniforme (capturou a estrutura forma->cor->posição)."""
    gd = _trained()
    T = gd.trans
    # alguma transição é forte (>0.5) e alguma é fraca (<0.2): estrutura, não uniforme (~0.33)
    assert T.max() > 0.5 and T.min() < 0.2
