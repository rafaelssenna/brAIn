"""Testes do M12 — escala + visão.

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import PredictiveCoder  # noqa: E402

SIDE = 12
D = SIDE * SIDE


def _patch(rng, n_edges=3):
    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    img = np.zeros((SIDE, SIDE))
    for _ in range(n_edges):
        th = rng.uniform(0, np.pi); cx, cy = rng.uniform(0, SIDE), rng.uniform(0, SIDE)
        proj = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        win = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (SIDE / 4.0) ** 2)))
        img += np.tanh(proj) * win * rng.uniform(0.5, 1.0)
    v = img.ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def _localization(W):
    k = max(1, D // 4)
    return float(np.mean([np.sort(W[:, i] ** 2)[::-1][:k].sum() / ((W[:, i] ** 2).sum() + 1e-12)
                          for i in range(W.shape[1])]))


def test_escala_aprende_entrada_144dim():
    """Escala: o predictive coder aprende em entrada visual de 144 dim."""
    rng = np.random.default_rng(0)
    pc = PredictiveCoder(n_obs=D, n_latent=48, eta_w=0.05, seed=0)
    patches = [_patch(rng) for _ in range(60)]
    e0 = np.mean([pc.prediction_error(p) for p in patches])
    for _ in range(3000):
        pc.learn(_patch(rng))
    e1 = np.mean([pc.prediction_error(p) for p in patches])
    assert e1 < 0.5 * e0


def test_atomos_ficam_localizados():
    """Os detectores aprendidos são LOCALIZADOS (não espalhados): marca de borda."""
    rng = np.random.default_rng(0)
    pc = PredictiveCoder(n_obs=D, n_latent=48, eta_w=0.05, seed=0)
    loc0 = _localization(pc.W)                 # W aleatório: espalhado (~0.25-0.4)
    for _ in range(4000):
        pc.learn(_patch(rng))
    loc1 = _localization(pc.W)
    assert loc1 > 0.5                          # claramente localizado
    assert loc1 > loc0                          # mais localizado que no nascimento
