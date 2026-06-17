"""Testes do M5 — garante que a curiosidade (learning progress) está correta.

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import IntrinsicMotivation, select_region  # noqa: E402


def test_lp_positivo_quando_erro_cai():
    """Erro caindo => learning progress > 0."""
    mot = IntrinsicMotivation(n_regions=1, tau_fast=5, tau_slow=30)
    for err in np.linspace(1.0, 0.1, 60):   # erro decrescente
        mot.update(0, err)
    assert mot.learning_progress()[0] > 0.0


def test_lp_quase_zero_quando_erro_estavel():
    """Erro estável (já dominado OU ruído) => learning progress ≈ 0."""
    mot = IntrinsicMotivation(n_regions=1, tau_fast=5, tau_slow=30)
    rng = np.random.default_rng(0)
    for _ in range(200):                    # erro alto e estável (tipo ruído)
        mot.update(0, 1.0 + 0.01 * rng.standard_normal())
    assert mot.learning_progress()[0] < 0.05


def test_curiosidade_evita_a_tv_chiando():
    """Entre uma região que melhora e uma de ruído (erro alto e fixo), a
    curiosidade prefere a que melhora; a novelty-seeking prefere o ruído."""
    mot = IntrinsicMotivation(n_regions=2, tau_fast=5, tau_slow=30)
    # região 0: aprendendo (erro cai); região 1: ruído (erro alto e fixo)
    for err in np.linspace(0.8, 0.2, 60):
        mot.update(0, err)
    for _ in range(60):
        mot.update(1, 1.5)
    rng = np.random.default_rng(0)
    # curiosidade (sem exploração) escolhe a região que progride
    cur = [select_region(mot, "curiosity", rng, eps=0.0) for _ in range(20)]
    assert all(c == 0 for c in cur)
    # novelty escolhe o ruído (maior erro) — a armadilha
    assert select_region(mot, "novelty", rng) == 1


def test_regiao_nao_visitada_tem_lp_infinito():
    """Otimismo: região nunca vista tem LP infinito (vale a pena explorar)."""
    mot = IntrinsicMotivation(n_regions=3)
    mot.update(0, 0.5)
    lp = mot.learning_progress()
    assert np.isinf(lp[1]) and np.isinf(lp[2])
    assert np.isfinite(lp[0])


def test_politica_random_cobre_todas():
    """A política aleatória eventualmente visita todas as regiões."""
    mot = IntrinsicMotivation(n_regions=4)
    rng = np.random.default_rng(0)
    chosen = {select_region(mot, "random", rng) for _ in range(200)}
    assert chosen == {0, 1, 2, 3}
