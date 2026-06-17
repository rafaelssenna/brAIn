"""Testes do M9 — predictive coding hierárquico.

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import HierarchicalPredictiveCoder  # noqa: E402


def _categories(C=3, exemplars=4, grid=8, seed=0):
    rng = np.random.default_rng(seed)
    D = grid * grid
    positions = [(1, 1), (1, 4), (4, 2)][:C]
    pats, labels = [], []
    for ci, (r0, c0) in enumerate(positions):
        base = np.zeros((grid, grid)); base[r0:r0 + 3, c0:c0 + 3] = 1.0
        for _ in range(exemplars):
            p = base.copy()
            for _ in range(2):
                rr, cc = rng.integers(grid), rng.integers(grid)
                p[rr, cc] = 1.0 - p[rr, cc]
            pats.append(p.ravel()); labels.append(ci)
    P = np.array(pats)
    return P / np.linalg.norm(P, axis=1, keepdims=True), np.array(labels), D


def _within_between(codes, labels):
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-9)
    S = codes @ codes.T
    w, b = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (w if labels[i] == labels[j] else b).append(S[i, j])
    return float(np.mean(w)), float(np.mean(b))


def test_hierarquia_reduz_erro():
    P, _, D = _categories()
    hpc = HierarchicalPredictiveCoder(sizes=[D, 16, 6], seed=0)
    rng = np.random.default_rng(0)
    e0 = hpc.learn(P[0])[1]
    errs = [hpc.learn(P[rng.integers(len(P))])[1] for _ in range(4000)]
    assert np.mean(errs[-200:]) < 0.5 * e0


def test_topo_forma_clusters_de_categoria():
    """No topo, exemplares da MESMA categoria são muito mais similares entre si
    do que de categorias diferentes (categorias viram clusters)."""
    P, labels, D = _categories()
    hpc = HierarchicalPredictiveCoder(sizes=[D, 16, 6], seed=0)
    rng = np.random.default_rng(0)
    for _ in range(6000):
        hpc.learn(P[rng.integers(len(P))] + 0.05 * rng.standard_normal(D))
    top = np.array([hpc.top_code(p) for p in P])
    w, b = _within_between(top, labels)
    assert w > 0.85           # invariância forte dentro da categoria
    assert w - b > 0.4        # categorias bem distintas no topo


def test_invariancia_cresce_da_entrada_ao_topo():
    """A similaridade DENTRO da categoria aumenta da entrada para o topo
    (a camada alta abstrai a variação dos exemplares)."""
    P, labels, D = _categories()
    hpc = HierarchicalPredictiveCoder(sizes=[D, 16, 6], seed=0)
    rng = np.random.default_rng(0)
    for _ in range(6000):
        hpc.learn(P[rng.integers(len(P))] + 0.05 * rng.standard_normal(D))
    w_in, _ = _within_between(P, labels)
    top = np.array([hpc.top_code(p) for p in P])
    w_top, _ = _within_between(top, labels)
    assert w_top > w_in       # mais invariante no topo


def test_reconstrucao_melhor_que_zero():
    P, _, D = _categories()
    hpc = HierarchicalPredictiveCoder(sizes=[D, 16, 6], seed=0)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        hpc.learn(P[rng.integers(len(P))])
    x = P[0]
    err_rec = np.sum((x - hpc.reconstruct(x)) ** 2)
    assert err_rec < np.sum(x ** 2)
