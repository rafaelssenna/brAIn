"""Testes do M1 — garante que a matemática do LIF está correta.

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LIFPopulation, fi_curve_theory  # noqa: E402


def test_repouso_sem_corrente():
    """Sem corrente, o potencial fica em repouso e não dispara."""
    neuron = LIFPopulation(n=1)
    spikes, voltages = neuron.run(np.zeros(1000))
    assert len(spikes[0]) == 0
    np.testing.assert_allclose(voltages[-1, 0], neuron.v_rest, atol=1e-6)


def test_corrente_sublimiar_nao_dispara():
    """Corrente abaixo do limiar (R*I < V_th - V_rest = 15 mV) não dispara."""
    neuron = LIFPopulation(n=1)  # R=10 MΩ -> I=1.0 nA dá só 10 mV
    spikes, _ = neuron.run(np.full(2000, 1.0))
    assert len(spikes[0]) == 0


def test_corrente_supralimiar_dispara():
    """Corrente acima do limiar gera disparos repetidos."""
    neuron = LIFPopulation(n=1)
    spikes, _ = neuron.run(np.full(2000, 2.0))  # 20 mV de drive
    assert len(spikes[0]) > 1


def test_periodo_refratario_respeitado():
    """Intervalos entre disparos nunca são menores que o período refratário."""
    neuron = LIFPopulation(n=1, refractory=5.0)
    spikes, _ = neuron.run(np.full(5000, 4.0))
    isi = np.diff(spikes[0])
    assert np.all(isi >= 5.0 - 1e-9)


def test_simulacao_bate_com_teoria():
    """A taxa simulada deve casar com a fórmula analítica (tolerância folgada
    por causa da discretização de Euler)."""
    dt = 0.05
    t_total = 2000.0
    n_steps = int(t_total / dt)
    for i_amp in [2.0, 3.0, 4.0]:
        neuron = LIFPopulation(n=1, dt=dt)
        spikes, _ = neuron.run(np.full(n_steps, i_amp), record_voltage=False)
        rate_sim = len(spikes[0]) / (t_total / 1000.0)
        rate_theory = fi_curve_theory(np.array([i_amp]))[0]
        assert abs(rate_sim - rate_theory) < 0.1 * rate_theory + 2.0


def test_vetorizacao_populacao():
    """N neurônios com a mesma corrente disparam de forma idêntica."""
    pop = LIFPopulation(n=5)
    currents = np.full((1000, 5), 2.5)
    spikes, _ = pop.run(currents, record_voltage=False)
    counts = [len(s) for s in spikes]
    assert len(set(counts)) == 1  # todos iguais
    assert counts[0] > 0
