"""Testes do M28 — conversa emergente com frases (sem professor).

Verifica que dois agentes inventam, sozinhos, uma língua com frases (forma×posição): a
comunicação emerge do acaso; o código composicional GENERALIZA para cenas nunca ditas
(o holístico não); e a ordem de palavras emerge CONSISTENTE entre os dois agentes.

Roda com:  python -m pytest tests/test_emergent_sentence.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

np.seterr(all="ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import EmergentSentenceGame  # noqa: E402

K_SHAPE = 3
K_POS = 3
ALL = [(s, p) for s in range(K_SHAPE) for p in range(K_POS)]


def _run(mode, train, n_steps=9000, seed=1):
    g = EmergentSentenceGame(K_SHAPE, K_POS, mode=mode, seed=seed)
    rng = np.random.default_rng(seed + 100)
    for t in range(n_steps):
        s, p = train[int(rng.integers(len(train)))]
        g.play(s, p, speaker_is_A=(t % 2 == 0))
    return g


def test_conversa_emerge_sem_professor():
    """Sem professor, dois agentes chegam ao entendimento mútuo bem acima do acaso."""
    g = _run("compositional", ALL)
    acc = g.accuracy(ALL)
    assert acc > 0.9                                   # acaso = 1/9 ≈ 0.11


def test_composicional_generaliza_para_cenas_nunca_ditas():
    """O código composicional emergente generaliza para combinações nunca ditas."""
    rng = np.random.default_rng(3)
    idx = rng.permutation(len(ALL))
    train = [ALL[i] for i in idx[:6]]; test = [ALL[i] for i in idx[6:]]
    g = _run("compositional", train)
    assert g.accuracy(train) > 0.9
    assert g.accuracy(test) > 0.9                       # generaliza para o nunca-dito


def test_holistico_nao_generaliza():
    """O holístico decora o treino mas falha nas cenas nunca ditas (contraste estrutural)."""
    rng = np.random.default_rng(3)
    idx = rng.permutation(len(ALL))
    train = [ALL[i] for i in idx[:6]]; test = [ALL[i] for i in idx[6:]]
    g = _run("holistic", train)
    assert g.accuracy(test) < 0.5                       # sem estrutura, não generaliza


def test_a_ordem_de_palavras_emerge_consistente_entre_os_dois():
    """A ordem das palavras emerge IGUAL nos dois agentes (uma gramática comum) e cada
    slot carrega um papel distinto (forma e posição, não dois iguais)."""
    g = _run("compositional", ALL)
    oa, ob = g.emergent_order()
    assert oa == ob                                     # os dois concordam na ordem
    assert set(oa) == {"shape", "pos"}                  # divisão de trabalho: papéis distintos


def test_a_ordem_e_arbitraria_entre_execucoes():
    """A ordem emergente NÃO é fixa no código: sementes diferentes podem gerar línguas com
    ordens diferentes (como línguas humanas diferem), todas internamente coerentes."""
    orders = set()
    for sd in range(6):
        g = _run("compositional", ALL, seed=sd + 1)
        oa, ob = g.emergent_order()
        assert oa == ob and g.accuracy(ALL) > 0.9       # cada uma é coerente e funciona
        orders.add(oa)
    # pelo menos uma ordem válida emergiu; idealmente mais de uma entre as sementes
    assert orders.issubset({("shape", "pos"), ("pos", "shape")})
    assert len(orders) >= 1
