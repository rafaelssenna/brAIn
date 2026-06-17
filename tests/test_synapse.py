"""Testes do M2 — garante que a regra STDP está correta.

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import STDPConnection  # noqa: E402


def _spike(mask_true_index, n):
    m = np.zeros(n, dtype=bool)
    m[mask_true_index] = True
    return m


def test_potenciacao_pre_antes_de_pos():
    """Pré dispara, depois (alguns passos) o pós dispara => peso AUMENTA."""
    conn = STDPConnection(1, 1, w_init=1.0, dt=1.0, tau_pre=20.0)
    pre, post = _spike(0, 1), _spike(0, 1)
    none = np.zeros(1, dtype=bool)

    conn.update(pre, none)            # t=0: pré dispara
    for _ in range(5):                # passa o tempo (5 ms)
        conn.update(none, none)
    conn.update(none, post)           # t=6: pós dispara

    assert conn.w[0, 0] > 1.0


def test_depressao_pos_antes_de_pre():
    """Pós dispara, depois o pré dispara => peso DIMINUI."""
    conn = STDPConnection(1, 1, w_init=1.0, dt=1.0, tau_post=20.0)
    pre, post = _spike(0, 1), _spike(0, 1)
    none = np.zeros(1, dtype=bool)

    conn.update(none, post)           # t=0: pós dispara
    for _ in range(5):
        conn.update(none, none)
    conn.update(pre, none)            # t=6: pré dispara

    assert conn.w[0, 0] < 1.0


def test_peso_limitado_em_zero_e_wmax():
    """Os pesos ficam presos no intervalo [0, w_max]."""
    conn = STDPConnection(1, 1, w_init=5.9, w_max=6.0, a_plus=1.0, dt=1.0)
    pre, post = _spike(0, 1), _spike(0, 1)
    none = np.zeros(1, dtype=bool)
    # Muita potenciação não passa de w_max.
    for _ in range(20):
        conn.update(pre, none)
        conn.update(none, post)
    assert conn.w[0, 0] <= 6.0 + 1e-9
    assert conn.w[0, 0] >= 0.0


def test_sinapse_fixa_nao_aprende():
    """plastic=False => peso nunca muda."""
    conn = STDPConnection(1, 1, w_init=2.0, plastic=False, dt=1.0)
    pre, post = _spike(0, 1), _spike(0, 1)
    conn.update(pre, post)
    assert conn.w[0, 0] == 2.0


def test_corrente_sinaptica_decai():
    """Após um disparo pré, a corrente injetada decai com o tempo."""
    conn = STDPConnection(1, 1, w_init=3.0, dt=1.0, tau_syn=10.0)
    pre = _spike(0, 1)
    none = np.zeros(1, dtype=bool)
    i0 = conn.propagate(pre)[0]
    i1 = conn.propagate(none)[0]
    i2 = conn.propagate(none)[0]
    assert i0 > i1 > i2 > 0.0
