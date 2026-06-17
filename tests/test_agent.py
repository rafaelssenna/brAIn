"""Testes do M7 — o organismo integrado (corpo + modelo de mundo + curiosidade).

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import RingWorld, IntegratedAgent  # noqa: E402


def test_mundo_lugar_aprendivel_e_consistente():
    """Lugar aprendível: padrão fixo (baixa variância entre visitas).
    Lugar de ruído: muda a cada visita (alta variância)."""
    w = RingWorld(n_locations=8, noise_locs=(3,), seed=1)
    rng = np.random.default_rng(0)
    a = w.sense(0, rng); b = w.sense(0, rng)          # lugar aprendível
    assert np.linalg.norm(a - b) < 0.8                 # parecido (só ruído pequeno ~0.5)
    c = w.sense(3, rng); d = w.sense(3, rng)          # lugar de ruído
    assert np.linalg.norm(c - d) > 1.0                 # bem diferente (~1.4)


def test_agente_aprende_modelo_de_mundo():
    """O domínio sobe: o erro de previsão nos lugares aprendíveis cai com a vida."""
    w = RingWorld(n_locations=6, noise_locs=(2,), seed=1)
    agent = IntegratedAgent(w, policy="curious", seed=0)
    rng = np.random.default_rng(0)
    learn = w.learnable_locs()
    err0 = np.mean([agent.pc.prediction_error(w.patterns[l]) for l in learn])
    for _ in range(3000):
        agent.step(rng)
    err1 = np.mean([agent.pc.prediction_error(w.patterns[l]) for l in learn])
    assert err1 < 0.5 * err0


def test_curioso_fica_menos_no_ruido_que_aleatorio():
    """O agente curioso passa menos tempo nos lugares de ruído que o passeio aleatório."""
    def noise_frac(policy, steps=3000):
        w = RingWorld(n_locations=8, noise_locs=(3, 6), seed=1)
        agent = IntegratedAgent(w, policy=policy, seed=0)
        rng = np.random.default_rng(0)
        cnt = 0
        for _ in range(steps):
            here, _ = agent.step(rng)
            cnt += int(here in (3, 6))
        return cnt / steps
    assert noise_frac("curious") < noise_frac("random")


def test_navegacao_anda_em_direcao_ao_alvo():
    """Com um alvo fixado, o agente dá um passo na direção mais curta do anel."""
    w = RingWorld(n_locations=8, noise_locs=(), seed=1)
    agent = IntegratedAgent(w, policy="curious", seed=0)
    agent.pos = 0
    nxt = agent._step_toward(2, 8)     # alvo à frente
    assert nxt == 1
    agent.pos = 0
    nxt = agent._step_toward(6, 8)     # alvo "atrás" (caminho mais curto = -1)
    assert nxt == 7
