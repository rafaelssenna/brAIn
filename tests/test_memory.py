"""Testes do M8 — memória / replay (consolidação contra esquecimento).

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import PredictiveCoder, ReplayBuffer  # noqa: E402


def test_buffer_respeita_capacidade():
    buf = ReplayBuffer(capacity=10, dim=4, seed=0)
    for i in range(100):
        buf.add(np.full(4, float(i)))
    assert buf.size == 10
    assert buf.n_seen == 100


def test_buffer_amostra_itens_vistos():
    buf = ReplayBuffer(capacity=50, dim=3, seed=0)
    vistos = set()
    for i in range(50):
        buf.add(np.full(3, float(i))); vistos.add(float(i))
    s = buf.sample(20)
    assert s.shape == (20, 3)
    assert all(v[0] in vistos for v in s)


def test_reservatorio_mantem_itens_antigos():
    """Amostragem por reservatório retém itens ANTIGOS (não só os recentes)."""
    buf = ReplayBuffer(capacity=100, dim=1, seed=0)
    for i in range(10000):
        buf.add(np.array([float(i)]))
    vals = buf.buf[:, 0]
    # deve haver itens da primeira metade da vida (um FIFO teria só os recentes)
    assert np.any(vals < 5000)


def test_replay_reduz_esquecimento():
    """O teste central da H4: reensaiar A enquanto aprende B retém A melhor."""
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (2, 16)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    B = rng.normal(0, 1, (3, 16)); B /= np.linalg.norm(B, axis=1, keepdims=True)

    def run(use_replay):
        pc = PredictiveCoder(n_obs=16, n_latent=10, eta_w=0.05, seed=0)
        buf = ReplayBuffer(200, 16, seed=0) if use_replay else None
        r = np.random.default_rng(1)
        for t in range(4000):
            src = A if t < 1500 else B
            x = src[r.integers(len(src))]
            pc.learn(x)
            if buf is not None:
                buf.add(x)
                if t % 5 == 0 and buf.size:
                    for xr in buf.sample(8):
                        pc.learn(xr)
        return float(np.mean([pc.prediction_error(a) for a in A]))   # erro final em A

    assert run(use_replay=True) < run(use_replay=False)
