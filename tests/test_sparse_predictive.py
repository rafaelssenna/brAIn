"""Testes do M22 — substrato esparso e dirigido a eventos.

Verifica: esparsidade respeita o orçamento k-WTA; soft-threshold (L1) zera; com
k=None,l1=0 degenera no M4; qualidade preservada dentro de tolerância; energia
(SynOps) reduzida por fator; campos receptivos ainda emergem; contador consistente.

Roda com:  python -m pytest tests/test_sparse_predictive.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import (SparsePredictiveCoder, PredictiveCoder, OpCounter,  # noqa: E402
                   dense_learn_macs)

GRID = 8
D = GRID * GRID
N = 16
NOISE = 0.06


def _objects():
    protos = []
    for row in range(0, GRID, 2):
        p = np.zeros((GRID, GRID)); p[row, :] = 1.0; protos.append(p)
    for col in range(0, GRID, 2):
        p = np.zeros((GRID, GRID)); p[:, col] = 1.0; protos.append(p)
    P = np.array([p.ravel() for p in protos])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def _train(coder, P, n_steps=5000, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    K = len(P)
    for _ in range(n_steps):
        coder.learn(P[rng.integers(K)] + NOISE * rng.standard_normal(D))
    return coder


def test_kwta_respeita_orcamento():
    """Com k-WTA, no máximo k unidades ficam ativas."""
    P = _objects()
    c = SparsePredictiveCoder(D, N, k_active=3, seed=0)
    _train(c, P, 2000)
    for o in range(len(P)):
        assert np.count_nonzero(c.infer(P[o])) <= 3


def test_l1_esparsifica_e_degenera():
    """L1 alto zera quase tudo; com l1=0 e k=None o coder é idêntico ao M4."""
    P = _objects()
    forte = SparsePredictiveCoder(D, N, l1=5.0, seed=0)
    _train(forte, P, 1500)
    assert np.count_nonzero(forte.infer(P[0])) < N            # L1 forte esparsifica

    # degeneração: mesma semente, mesmo fluxo => W idêntico ao M4
    rng_a = np.random.default_rng(3); rng_b = np.random.default_rng(3)
    m4 = PredictiveCoder(D, N, seed=0)
    sp = SparsePredictiveCoder(D, N, k_active=None, l1=0.0, seed=0)
    for _ in range(300):
        xa = P[rng_a.integers(len(P))] + NOISE * rng_a.standard_normal(D)
        xb = P[rng_b.integers(len(P))] + NOISE * rng_b.standard_normal(D)
        m4.learn(xa); sp.learn(xb)
    assert np.allclose(m4.W, sp.W, atol=1e-9)


def test_qualidade_preservada():
    """O esparso (k-WTA) atinge qualidade de previsão dentro de tolerância do denso."""
    P = _objects()
    rng = np.random.default_rng(7)
    dense = _train(SparsePredictiveCoder(D, N, eta_w=0.05, k_active=None, seed=0), P, 6000)
    sparse = _train(SparsePredictiveCoder(D, N, eta_w=0.05, k_active=2, seed=0), P, 6000)
    hd = np.mean([dense.prediction_error(P[rng.integers(len(P))] + NOISE * rng.standard_normal(D))
                  for _ in range(2000)])
    hs = np.mean([sparse.prediction_error(P[rng.integers(len(P))] + NOISE * rng.standard_normal(D))
                  for _ in range(2000)])
    assert hs < 1.6 * hd                  # esparso pode ser levemente pior, não muito
    assert hs < 0.25 * np.sum(P[0] ** 2)  # e aprendeu de verdade (bem abaixo de prever nada)


def test_energia_reduzida_por_fator():
    """O k-WTA gasta bem menos operações sinápticas que o denso pleno."""
    P = _objects()
    sparse = _train(SparsePredictiveCoder(D, N, k_active=2, seed=0), P, 3000)
    cnt = OpCounter()
    for o in range(len(P)):
        sparse.infer(P[o], counter=cnt)
    syn_sparse = cnt.total_warm() / len(P)
    syn_dense = dense_learn_macs(D, N, 30)
    assert syn_sparse < syn_dense / 3.0   # pelo menos 3x menos (N/k = 8)


def test_estrutura_emerge_esparsa():
    """Mesmo esparso, os campos receptivos recuperam as barras do mundo."""
    P = _objects()
    c = _train(SparsePredictiveCoder(D, N, eta_w=0.05, k_active=2, seed=0), P, 7000)
    Wn = c.W / (np.linalg.norm(c.W, axis=0, keepdims=True) + 1e-9)
    for o in range(len(P)):
        cos = np.abs(P[o] @ Wn)
        assert cos.max() > 0.9            # alguma coluna casa com cada barra


def test_contador_consistente():
    """total = soma das parcelas; reset zera; warm <= full."""
    P = _objects()
    c = SparsePredictiveCoder(D, N, k_active=2, seed=0)
    cnt = OpCounter()
    c.learn(P[0], counter=cnt)
    assert cnt.total_full() == cnt.macs_predict + cnt.macs_bottomup_full + cnt.macs_update
    assert cnt.total_warm() <= cnt.total_full()      # dirigido a eventos <= piso denso
    cnt.reset()
    assert cnt.total_full() == 0


def test_discrimina_conceitos():
    """O código esparso ainda separa os K padrões (conceitos distintos)."""
    P = _objects()
    c = _train(SparsePredictiveCoder(D, N, eta_w=0.05, k_active=2, seed=0), P, 7000)
    cs = [int(np.argmax(c.infer(P[o]))) for o in range(len(P))]
    assert len(set(cs)) >= len(P) - 1                # distingue ao menos K-1 dos K
