"""Testes do M10 — predictive coding spiking (substrato unificado).

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import SpikingPredictiveCoder  # noqa: E402


def _patterns(n=3, d=16, seed=1):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (n, d))
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def test_rede_spiking_aprende():
    """O teste central do M10: a surpresa cai treinando numa rede spiking."""
    P = _patterns(3, 16)
    spc = SpikingPredictiveCoder(n_obs=16, n_latent=6, seed=0)
    rng = np.random.default_rng(0)
    e0 = np.mean([spc.prediction_error(p) for p in P])
    for _ in range(2000):
        spc.learn(P[rng.integers(len(P))] + 0.05 * rng.standard_normal(16))
    e1 = np.mean([spc.prediction_error(p) for p in P])
    assert e1 < 0.4 * e0


def test_inferencia_assenta_sem_oscilar():
    """A inferência amortecida reduz o erro vs r=0 (não oscila de volta ao topo)."""
    P = _patterns(3, 16)
    spc = SpikingPredictiveCoder(n_obs=16, n_latent=6, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(800):
        spc.learn(P[rng.integers(len(P))])
    x = P[0]
    err_zero = float(np.sum(x ** 2))            # erro com rho=0
    err_inf = spc.prediction_error(x)           # erro após inferência spiking
    assert err_inf < err_zero


def test_taxas_de_disparo_normalizadas():
    """As taxas de disparo (ρ) ficam em [0, 1] e são não-negativas."""
    spc = SpikingPredictiveCoder(n_obs=16, n_latent=6, seed=0)
    rho, _ = spc.infer(_patterns()[0])
    assert np.all(rho >= 0.0) and np.all(rho <= 1.0)


def test_corrente_negativa_silencia():
    """Corrente negativa => neurônio não dispara (retificação pelo limiar)."""
    spc = SpikingPredictiveCoder(n_obs=16, n_latent=6, seed=0)
    rates = spc._spike_rates(np.full(6, -3.0))   # corrente bem negativa
    assert np.all(rates == 0.0)
