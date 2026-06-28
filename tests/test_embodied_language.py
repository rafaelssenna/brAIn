"""Testes do M25 — o organismo corporificado que fala.

Verifica que o organismo COMPLETO (corpo + curiosidade + percepção + memória +
linguagem) vive num laço só: navega o mundo, aprende a perceber, se encontra com o
outro e a linguagem emerge — corporificada, com o gargalo do encontro.

Roda com:  python -m pytest tests/test_embodied_language.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

np.seterr(all="ignore")     # warnings espúrios de matmul (NumPy/BLAS); resultados são finitos

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import RingWorld, EmbodiedLanguageAgent, communication_accuracy  # noqa: E402


def _world():
    return RingWorld(n_locations=8, grid=8, noise_locs=(3, 6), seed=0)


def _live(world, n_steps, spiking=False, seed_rng=42):
    learn_locs = world.learnable_locs()
    clean = [world.patterns[l] for l in learn_locs]
    K = len(learn_locs)
    A = EmbodiedLanguageAgent(world, 12, K, eps=0.25, dwell=6, spiking=spiking, seed=1)
    B = EmbodiedLanguageAgent(world, 12, K, eps=0.25, dwell=6, spiking=spiking, seed=2)
    rng = np.random.default_rng(seed_rng)
    encounters = 0
    for t in range(n_steps):
        A.perceive_and_learn(rng); B.perceive_and_learn(rng)
        if A.pos == B.pos and A.pos in learn_locs:
            encounters += 1
            obj = world.patterns[A.pos]
            spk, lis = (A, B) if t % 2 == 0 else (B, A)
            c = spk.concept_of(obj); m = spk.speak(c)
            reward = (lis.listen(m) == lis.concept_of(obj))
            spk.reinforce_speak(c, m, reward)
            lis.learn_listen(m, lis.concept_of(obj))
        A.navigate(); B.navigate()
    return A, B, clean, encounters


def test_organismo_corporificado_percebe_e_fala():
    """Vivendo no corpo, o organismo aprende a perceber e a linguagem emerge acima do acaso."""
    world = _world()
    A, B, clean, enc = _live(world, n_steps=6000)
    K = len(clean)
    # percebe: distingue a maioria dos objetos
    assert len(set(A.concept_of(clean[j]) for j in range(K))) / K > 0.8
    # fala: comunicação acima do acaso (1/K)
    comm = 0.5 * (communication_accuracy(A, B, clean) + communication_accuracy(B, A, clean))
    assert comm > 1.5 / K


def test_o_corpo_se_move_e_se_encontra():
    """O corpo navega o mundo (não fica parado) e os agentes se encontram."""
    world = _world()
    A, B, _, enc = _live(world, n_steps=3000)
    assert enc > 0                                   # houve atenção conjunta corporificada
    visited = set()
    rng = np.random.default_rng(0)
    for _ in range(200):
        visited.add(A.pos); A.navigate()
    assert len(visited) > 1                          # o corpo de fato se move


def test_surpresa_cai_vivendo():
    """A percepção aprende vivendo: a surpresa nos objetos cai."""
    world = _world()
    learn_locs = world.learnable_locs()
    clean = [world.patterns[l] for l in learn_locs]
    A = EmbodiedLanguageAgent(world, 12, len(learn_locs), eps=0.25, dwell=6, seed=1)
    rng = np.random.default_rng(0)
    e0 = np.mean([A.surprise_on(c) for c in clean])
    for _ in range(2000):
        A.perceive_and_learn(rng); A.navigate()
    e1 = np.mean([A.surprise_on(c) for c in clean])
    assert e1 < 0.5 * e0


def test_comunicacao_e_limitada_pelo_encontro():
    """O gargalo da corporificação: com pouquíssimos encontros, a língua não alinha
    tão bem quanto com muitos (a comunicação acompanha a atenção conjunta)."""
    world = _world()
    _, _, clean, enc_short = _live(world, n_steps=1500)
    A_s, B_s, _, _ = _live(world, n_steps=1500)
    A_l, B_l, _, enc_long = _live(world, n_steps=7000)
    comm_short = 0.5 * (communication_accuracy(A_s, B_s, clean) + communication_accuracy(B_s, A_s, clean))
    comm_long = 0.5 * (communication_accuracy(A_l, B_l, clean) + communication_accuracy(B_l, A_l, clean))
    assert enc_long > enc_short
    assert comm_long >= comm_short                    # mais encontros, língua não pior


def test_corporificado_compoe_com_spiking():
    """O organismo corporificado roda também com percepção spiking (M24) — compõe."""
    world = _world()
    A, B, clean, enc = _live(world, n_steps=1200, spiking=True)
    from brain.spiking_predictive import SpikingPerceptionCoder
    assert isinstance(A.mind.pc, SpikingPerceptionCoder)
    assert enc > 0
