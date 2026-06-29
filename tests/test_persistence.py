"""Testes de persistência — salvar/carregar o cérebro (memória de longo prazo).

Verifica que o que o agente APRENDEU sobrevive a salvar e recarregar: ele acorda com o
mesmo conhecimento, não renasce bebê.

Roda com:  python -m pytest tests/test_persistence.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

np.seterr(all="ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LivingAgent  # noqa: E402

SIDE = 8
D = SIDE * SIDE
K = 6


def _objects():
    pats = []
    for r in [1, 4, 6]:
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    for c in [1, 4, 6]:
        p = np.zeros((SIDE, SIDE)); p[:, c] = 1.0; pats.append(p.ravel())
    P = np.array(pats)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


P = _objects()


def _train(agent, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        i = int(rng.integers(K))
        obj = P[i] + 0.06 * rng.standard_normal(D); obj /= np.linalg.norm(obj)
        c = agent.concept(obj)
        agent.learn_perception(obj); agent.learn_listen(i, c); agent.reinforce_speak(c, i, True)
    return agent


def _naming(agent):
    return np.mean([agent.speak(agent.concept(P[i]), explore=False) == i for i in range(K)])


def test_salvar_e_carregar_preserva_o_aprendido():
    """Um agente treinado, salvo e recarregado, mantém a acurácia de nomear."""
    a = _train(LivingAgent(D, 12, K, seed=1))
    before = _naming(a)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "brain.npz")
        a.save(path)
        b = LivingAgent.load(path)
    assert _naming(b) == before                     # acorda com o mesmo conhecimento


def test_pesos_identicos_apos_carregar():
    """Os pesos e matrizes carregados são idênticos aos salvos."""
    a = _train(LivingAgent(D, 12, K, seed=1))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "brain.npz")
        a.save(path)
        b = LivingAgent.load(path)
    assert np.allclose(a.pc.W, b.pc.W)
    assert np.allclose(a.S, b.S) and np.allclose(a.R, b.R)
    assert b._t == a._t                             # também lembra quanto já viveu


def test_agente_novo_nasce_sem_saber():
    """Contraste: um agente novo (não carregado) nomeia ~ao acaso (nasce burro)."""
    fresh = LivingAgent(D, 12, K, seed=7)
    assert _naming(fresh) < 0.5                      # acaso = 1/6 ≈ 0.17


def test_continuar_treino_depois_de_carregar():
    """Carregar e continuar treinando funciona (o aprendizado acumula, não reinicia)."""
    a = _train(LivingAgent(D, 12, K, seed=1), n=500)   # treino curto -> imperfeito
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "brain.npz")
        a.save(path)
        b = LivingAgent.load(path)
        _train(b, n=2500, seed=1)                       # continua treinando
    assert _naming(b) >= _naming(a)                     # ficou pelo menos tão bom (acumulou)


def test_load_aceita_caminho_com_ou_sem_extensao():
    """load() funciona com o caminho com .npz ou sem (conveniência)."""
    a = _train(LivingAgent(D, 12, K, seed=1), n=300)
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "brain")
        a.save(base + ".npz")
        b = LivingAgent.load(base)                      # sem extensão
    assert b.M == K and b.pc.W.shape == a.pc.W.shape
