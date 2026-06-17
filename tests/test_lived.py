"""Testes do M19 — a língua nasce da vivência (conceitos auto-aprendidos).

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import PredictiveCoder, LivedLanguageGame  # noqa: E402

SIDE = 8
D = SIDE * SIDE
K = 8


def _objects():
    pats = []
    for r in range(0, SIDE, 2):
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    for c in range(0, SIDE, 2):
        p = np.zeros((SIDE, SIDE)); p[:, c] = 1.0; pats.append(p.ravel())
    P = np.array(pats[:K])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def _learn(P, n_latent, seed, steps=4000):
    pc = PredictiveCoder(n_obs=D, n_latent=n_latent, eta_w=0.05, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        pc.learn(P[rng.integers(K)] + 0.06 * rng.standard_normal(D))
    return np.array([int(np.argmax(pc.infer(P[i]))) for i in range(K)])


def _comm(cmapA, cmapB, seed, rounds=6000):
    g = LivedLanguageGame(cmapA, cmapB, n_symbols=K, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        g.play(int(rng.integers(K)))
    return g


def test_lingua_emerge_de_conceitos_aprendidos():
    """Dois agentes que aprenderam a distinguir tudo desenvolvem uma língua que funciona."""
    P = _objects()
    cmA = _learn(P, 12, 1); cmB = _learn(P, 12, 2)
    g = _comm(cmA, cmB, seed=3)
    assert g.receiver_discrim() > 0.9        # B aprendeu a distinguir os objetos
    assert g.accuracy() > 0.85               # a língua funciona


def test_concepto_pobre_limita_a_comunicacao():
    """Se B aprendeu conceitos grosseiros (poucos latentes), ele funde objetos e a
    comunicação cai — só se fala do que ambos distinguem."""
    P = _objects()
    cmA = _learn(P, 12, 1)
    cmB_poor = _learn(P, 3, 7)               # capacidade insuficiente -> funde objetos
    g = _comm(cmA, cmB_poor, seed=5)
    assert g.receiver_discrim() < 1.0
    assert g.accuracy() <= g.ceiling() + 0.15


def test_metricas_de_alinhamento():
    """Teto e discriminabilidade refletem conceitos idênticos vs. fundidos."""
    ident = np.arange(K)                     # cada objeto, um conceito único
    merged = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    g1 = LivedLanguageGame(ident, ident, n_symbols=K)
    assert g1.ceiling() == 1.0 and g1.receiver_discrim() == 1.0
    g2 = LivedLanguageGame(ident, merged, n_symbols=K)
    assert g2.ceiling() == 0.0               # B não dá conceito único a nenhum objeto
    assert g2.receiver_discrim() < 1.0
