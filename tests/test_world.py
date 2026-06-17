"""Testes do M3 — garante que o mundo e o corpo se comportam corretamente.

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LightWorld, Vehicle  # noqa: E402


def test_intensidade_cai_com_distancia():
    w = LightWorld(size=100.0, falloff=40.0)
    assert w.intensity(0.0) > w.intensity(40.0) > w.intensity(100.0)
    assert w.intensity(0.0) == 1.0


def test_sensor_ve_mais_do_lado_da_fonte():
    """Fonte à esquerda do agente => olho esquerdo ativa mais que o direito."""
    w = LightWorld(size=100.0, source=(50.0, 80.0))  # fonte ao norte
    pos = np.array([50.0, 50.0])
    heading = 0.0  # olhando para +x (leste); a fonte está à esquerda (norte)
    left, right, dist = w.sensor_activations(pos, heading)
    assert left > right
    assert np.isclose(dist, 30.0)


def test_sensor_simetrico_com_fonte_a_frente():
    """Fonte exatamente à frente => olhos esquerdo e direito iguais."""
    w = LightWorld(size=100.0, source=(80.0, 50.0))  # fonte a leste
    left, right, _ = w.sensor_activations(np.array([50.0, 50.0]), 0.0)
    assert np.isclose(left, right)


def test_veiculo_anda_reto_com_rodas_iguais():
    v = Vehicle(pos=(50.0, 50.0), heading=0.0)
    h0 = v.heading
    v.step(2.0, 2.0)
    assert np.isclose(v.heading, h0)         # não gira
    assert v.pos[0] > 50.0                    # avançou em +x
    assert np.isclose(v.pos[1], 50.0)


def test_veiculo_gira_com_rodas_diferentes():
    """Roda direita mais rápida => vira anti-horário (heading aumenta)."""
    v = Vehicle(pos=(50.0, 50.0), heading=0.0)
    v.step(0.0, 2.0)
    assert v.heading > 0.0


def test_veiculo_fica_na_arena():
    v = Vehicle(pos=(98.0, 98.0), heading=0.0, size=100.0)
    for _ in range(50):
        v.step(5.0, 5.0)
    assert 0.0 <= v.pos[0] <= 100.0
    assert 0.0 <= v.pos[1] <= 100.0


def test_distancia_ate_alvo():
    v = Vehicle(pos=(0.0, 0.0), heading=0.0)
    assert np.isclose(v.distance_to(np.array([3.0, 4.0])), 5.0)
