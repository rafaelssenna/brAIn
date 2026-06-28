"""Testes do M24 — o organismo vivo (M20) percebendo com neurônios que DISPARAM.

Verifica que a cognição (percepção + linguagem) sobrevive quando a percepção é feita
pelo substrato spiking (neurônios LIF, M10) dentro do organismo, que a esparsidade é
EMERGENTE (nem todo neurônio dispara), e que `spiking=False` não regride o M20.

Roda com:  python -m pytest tests/test_spiking_organism.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

np.seterr(all="ignore")     # warnings espúrios de matmul (NumPy/BLAS); resultados são finitos

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LivingAgent, OpCounter  # noqa: E402
from brain.spiking_predictive import SpikingPerceptionCoder  # noqa: E402

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


def _live(spiking, P, n_steps, rng_seed=3):
    A = LivingAgent(D, N_LATENT, 8, seed=1, spiking=spiking)
    B = LivingAgent(D, N_LATENT, 8, seed=2, spiking=spiking)
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


def test_cognicao_sobrevive_ao_spiking():
    """Percebendo com neurônios que disparam, percepção e linguagem ainda co-emergem."""
    P = _objects()
    A, B = _live(True, P, n_steps=900)
    assert A.discriminability(P) > 0.8                          # ainda percebe
    assert 0.5 * (_comm(A, B, P) + _comm(B, A, P)) > 0.45        # ainda fala (acaso ~0.17)


def test_substrato_e_spiking_de_verdade():
    """O substrato é o coder spiking, e a percepção aprende (surpresa cai)."""
    P = _objects()
    a = LivingAgent(D, N_LATENT, 8, seed=1, spiking=True)
    assert isinstance(a.pc, SpikingPerceptionCoder)
    rng = np.random.default_rng(0)
    e0 = np.mean([a.pc.prediction_error(P[j]) for j in range(K)])
    for _ in range(300):
        oi = int(rng.integers(K)); a.learn_perception(P[oi] + 0.06 * rng.standard_normal(D))
    e1 = np.mean([a.pc.prediction_error(P[j]) for j in range(K)])
    assert e1 < 0.5 * e0                                        # aprendeu a prever (surpresa caiu)


def test_esparsidade_emergente_do_limiar():
    """Nem todo neurônio dispara: a esparsidade EMERGE do limiar do spike (não imposta)."""
    P = _objects()
    a, _ = _live(True, P, n_steps=300)
    af = a.pc.active_fraction(P[0])
    assert 0.0 < af < 1.0                                       # alguns disparam, nem todos


def test_synops_contabilizadas():
    """A inferência spiking contabiliza operações sinápticas via OpCounter."""
    P = _objects()
    a = LivingAgent(D, N_LATENT, 8, seed=1, spiking=True)
    cnt = OpCounter()
    a.pc.infer(P[0], counter=cnt)
    assert cnt.total_warm() > 0


def test_spiking_false_nao_regride_m20():
    """spiking=False mantém o organismo idêntico ao M20 (substrato denso/esparso)."""
    from brain.sparse_predictive import SparsePredictiveCoder
    a = LivingAgent(D, N_LATENT, 8, seed=1, spiking=False)
    assert isinstance(a.pc, SparsePredictiveCoder)
    assert not isinstance(a.pc, SpikingPerceptionCoder)
