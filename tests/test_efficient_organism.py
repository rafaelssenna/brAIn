"""Testes do M23 — o organismo vivo (M20) rodando no substrato esparso (M22).

Verifica que a cognição (percepção + linguagem) sobrevive à esparsidade e que o
substrato esparso gasta menos operações sinápticas.

Roda com:  python -m pytest tests/test_efficient_organism.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LivingAgent, OpCounter  # noqa: E402

SIDE = 8
D = SIDE * SIDE
K = 6
N_LATENT = 16


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


def _live(sparse_k, P, n_steps=4500, rng_seed=3):
    A = LivingAgent(D, N_LATENT, 8, seed=1, sparse_k=sparse_k)
    B = LivingAgent(D, N_LATENT, 8, seed=2, sparse_k=sparse_k)
    rng = np.random.default_rng(rng_seed)
    for t in range(n_steps):
        oi = int(rng.integers(K)); obj = P[oi] + 0.06 * rng.standard_normal(D)
        A.learn_perception(obj); B.learn_perception(obj)
        spk, lis = (A, B) if t % 2 == 0 else (B, A)
        c = spk.concept(obj); m = spk.speak(c)
        lc = [lis.concept(P[j]) for j in range(K)]
        cands = [j for j in range(K) if lc[j] == lis.listen(m)]
        guess = int(rng.choice(cands)) if cands else int(rng.integers(K))
        spk.reinforce_speak(c, m, guess == oi)
        lis.learn_listen(m, lis.concept(obj))
    return A, B


def test_cognicao_sobrevive_a_esparsidade():
    """No substrato esparso, percepção e linguagem ainda co-emergem bem."""
    P = _objects()
    A, B = _live(2, P)
    assert A.discriminability(P) > 0.8                       # ainda percebe
    assert 0.5 * (_comm(A, B, P) + _comm(B, A, P)) > 0.45     # ainda fala (acaso ~0.17)


def test_organismo_esparso_gasta_menos_energia():
    """O substrato esparso usa menos operações sinápticas por percepção que o denso."""
    P = _objects()
    A_dense, _ = _live(None, P, n_steps=2500)
    A_spar, _ = _live(2, P, n_steps=2500)

    def synops(agent):
        cnt = OpCounter()
        for o in range(K):
            agent.pc.infer(P[o], counter=cnt)
        return cnt.total_warm() / K

    assert synops(A_spar) < synops(A_dense)
    assert A_spar.pc.active_fraction(P[0]) <= 2 / N_LATENT + 1e-9   # respeita o orçamento
