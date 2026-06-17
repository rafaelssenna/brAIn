"""Testes do M15 — gramática (aprender regras sequenciais e gerar frases válidas).

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import TemporalPredictiveCoder  # noqa: E402

D = 16
K = 6
TEMP = 0.12

T = np.zeros((K, K))
T[0, 1] = 0.5; T[0, 2] = 0.5
T[1, 3] = 1.0; T[2, 3] = 1.0
T[3, 4] = 0.6; T[3, 5] = 0.4
T[4, 0] = 1.0; T[5, 0] = 1.0


def _protos(seed=1):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (K, D))
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def _sample(n, rng, start=0):
    s = start; out = [s]
    for _ in range(n - 1):
        s = int(rng.choice(K, p=T[s])); out.append(s)
    return out


def _dist(model, protos):
    sc = protos @ model.predict()
    z = np.exp((sc - sc.max()) / TEMP)
    return z / z.sum()


def _train():
    protos = _protos()
    model = TemporalPredictiveCoder(D, decay=0.25, eta_w=0.05, seed=0)
    rng = np.random.default_rng(42)
    seq = _sample(12000, rng)
    model.reset_context()
    for t in range(len(seq) - 1):
        model.observe(protos[seq[t]] + 0.02 * rng.standard_normal(D))
    return model, protos


def test_aprende_matriz_de_transicao():
    """A gramática aprendida (distribuição prevista por símbolo) bate com a real."""
    model, protos = _train()
    rng = np.random.default_rng(1)
    seq = _sample(3000, rng)
    rows = np.zeros((K, K)); cnt = np.zeros(K)
    model.reset_context()
    for t in range(len(seq) - 1):
        model.observe(protos[seq[t]], learn=False)
        rows[seq[t]] += _dist(model, protos); cnt[seq[t]] += 1
    learned = rows / np.maximum(cnt, 1)[:, None]
    assert np.mean(np.abs(T - learned)) < 0.12


def test_gera_sequencias_gramaticais():
    """As sequências geradas respeitam a gramática muito mais que o acaso."""
    model, protos = _train()
    g = np.random.default_rng(7)
    model.reset_context(); model.observe(protos[0], learn=False)
    seq = [0]
    for _ in range(300):
        nxt = int(g.choice(K, p=_dist(model, protos)))
        seq.append(nxt); model.observe(protos[nxt], learn=False)
    gram = np.mean([T[seq[t], seq[t + 1]] > 0 for t in range(len(seq) - 1)])
    rand = list(np.random.default_rng(3).integers(K, size=300))
    rand_gram = np.mean([T[rand[t], rand[t + 1]] > 0 for t in range(len(rand) - 1)])
    assert gram > 0.85
    assert gram > 2 * rand_gram
