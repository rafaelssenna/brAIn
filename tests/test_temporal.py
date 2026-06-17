"""Testes do M14 — previsão temporal/sequencial.

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import TemporalPredictiveCoder  # noqa: E402

D = 16


def _symbols(k, seed=1):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (k, D))
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def _train(seq, P, decay, reps, seed=0):
    tpc = TemporalPredictiveCoder(D, decay=decay, seed=0)
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(reps):
        for i in seq:
            e, _ = tpc.observe(P[i] + 0.02 * rng.standard_normal(D))
            errs.append(e)
    return tpc, np.array(errs)


def test_aprende_sequencia_estruturada():
    """O erro cai numa sequência repetida (aprendeu a prever o próximo)."""
    P = _symbols(5)
    _, errs = _train([0, 1, 2, 3, 4], P, decay=0.5, reps=300)
    assert np.mean(errs[-100:]) < 0.3 * np.mean(errs[:100])


def test_nao_aprende_sequencia_aleatoria():
    """Controle: numa sequência aleatória o erro NÃO cai (não há estrutura)."""
    P = _symbols(5)
    rng = np.random.default_rng(0)
    seq = list(rng.integers(5, size=1500))
    _, errs = _train(seq, P, decay=0.5, reps=1)
    assert np.mean(errs[-200:]) > 0.6 * np.mean(errs[:200])


def test_imagina_a_continuacao():
    """Treinado num ciclo, gera a continuação correta a partir de uma semente."""
    P = _symbols(5)
    tpc, _ = _train([0, 1, 2, 3, 4], P, decay=0.5, reps=500)
    tpc.reset_context()
    for i in [0, 1, 2]:
        tpc.observe(P[i], learn=False)
    gen = tpc.generate(6, P)
    assert gen == [3, 4, 0, 1, 2, 3]


def test_memoria_ajuda_em_sequencia_contextual():
    """Quando o próximo depende do passado, COM memória erra menos que SEM."""
    P = _symbols(4, seed=2)
    seq = [0, 1, 0, 2]                       # após 0 vem 1 ou 2 (depende do contexto)
    _, e_mem = _train(seq, P, decay=0.6, reps=500, seed=7)
    _, e_no = _train(seq, P, decay=0.0, reps=500, seed=7)
    assert np.mean(e_mem[-200:]) < np.mean(e_no[-200:])
