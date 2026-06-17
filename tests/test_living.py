"""Testes do M20 — o organismo vivo (percepção + memória + linguagem num laço).

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LivingAgent  # noqa: E402

SIDE = 8
D = SIDE * SIDE
K = 6


def _objects():
    pats = []
    for r in range(0, SIDE, 2):
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    for c in range(0, SIDE, 3):
        p = np.zeros((SIDE, SIDE)); p[:, c] = 1.0; pats.append(p.ravel())
    P = np.array(pats[:K])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def _comm(spk, lis, P):
    lc = [lis.concept(P[j]) for j in range(len(P))]
    tot = 0.0
    for o in range(len(P)):
        m = spk.speak(spk.concept(P[o]), explore=False)
        cands = [j for j in range(len(P)) if lc[j] == lis.listen(m)]
        if o in cands:
            tot += 1.0 / len(cands)
    return tot / len(P)


def test_agente_vivo_aprende_a_perceber():
    """Vivendo o mundo, o agente passa a distinguir os objetos."""
    P = _objects()
    a = LivingAgent(D, 10, 8, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(3000):
        a.learn_perception(P[rng.integers(K)] + 0.06 * rng.standard_normal(D))
    assert a.discriminability(P) > 0.8


def test_percepcao_e_linguagem_co_emergem():
    """No mesmo laço, dois agentes aprendem a ver E a falar: comunicação bem acima do acaso."""
    P = _objects()
    A = LivingAgent(D, 10, 8, seed=1); B = LivingAgent(D, 10, 8, seed=2)
    rng = np.random.default_rng(3)
    for t in range(4500):
        oi = int(rng.integers(K)); obj = P[oi] + 0.06 * rng.standard_normal(D)
        A.learn_perception(obj); B.learn_perception(obj)
        spk, lis = (A, B) if t % 2 == 0 else (B, A)
        c = spk.concept(obj); m = spk.speak(c)
        lc = [lis.concept(P[j]) for j in range(K)]
        cands = [j for j in range(K) if lc[j] == lis.listen(m)]
        guess = int(rng.choice(cands)) if cands else int(rng.integers(K))
        spk.reinforce_speak(c, m, guess == oi)
        lis.learn_listen(m, lis.concept(obj))
    assert A.discriminability(P) > 0.8                 # aprendeu a ver
    assert 0.5 * (_comm(A, B, P) + _comm(B, A, P)) > 0.45   # e a falar (acaso ~0.17)
