"""Testes do M21 — aprender palavras reais de português ancoradas na percepção.

Roda com:  python -m pytest tests/test_portuguese.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LivingAgent  # noqa: E402

SIDE = 8
D = SIDE * SIDE
ROWS = [1, 4, 6]
COLS = [1, 4, 6]


def _objects():
    pats = []
    for r in ROWS:
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    for c in COLS:
        p = np.zeros((SIDE, SIDE)); p[:, c] = 1.0; pats.append(p.ravel())
    P = np.array(pats)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def _teach(n_steps=4000, n_latent=12, noise=0.06, seed=1, rng_seed=7):
    P = _objects(); K = len(P)
    agent = LivingAgent(D, n_latent, K, seed=seed)
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_steps):
        o = int(rng.integers(K))
        obj = P[o] + noise * rng.standard_normal(D)
        agent.learn_perception(obj)
        c = agent.concept(obj)
        agent.learn_listen(o, c)
        agent.reinforce_speak(c, o, True)
    return agent, P


def _naming(agent, P):
    return sum(agent.speak(agent.concept(P[o]), explore=False) == o for o in range(len(P))) / len(P)


def _comprehension(agent, P):
    cs = [agent.concept(P[j]) for j in range(len(P))]
    tot = 0.0
    for w in range(len(P)):
        cands = [j for j in range(len(P)) if cs[j] == agent.listen(w)]
        if w in cands:
            tot += 1.0 / len(cands)
    return tot / len(P)


def test_aprende_a_nomear_em_portugues():
    """Depois de viver ouvindo as palavras, ele nomeia certo o que vê."""
    agent, P = _teach()
    assert _naming(agent, P) >= 0.8


def test_aprende_a_compreender_em_portugues():
    """E aponta o objeto certo quando ouve a palavra."""
    agent, P = _teach()
    assert _comprehension(agent, P) >= 0.8


def test_lingua_limitada_pela_percepcao():
    """Com percepção pobre (poucos latentes), ele não distingue tudo e erra nomes:
    a linguagem é limitada pelo que ele consegue perceber."""
    P = _objects()
    fraco, _ = _teach(n_steps=2500, n_latent=2)      # quase não distingue os objetos
    forte, _ = _teach(n_steps=2500, n_latent=12)     # distingue bem
    assert _naming(fraco, P) < _naming(forte, P)
