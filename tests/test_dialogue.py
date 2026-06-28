"""Testes do M29 — diálogo com turnos (pergunta → resposta ancorada).

Verifica que o agente aprende a RESPONDER À PERGUNTA: usar o tipo de pergunta para escolher
o que descrever (forma OU posição). O agente CEGO à pergunta acerta a forma mas falha a
posição — prova que usar a pergunta é o que faz o diálogo funcionar.

Roda com:  python -m pytest tests/test_dialogue.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

np.seterr(all="ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import DialogueGame, ASK_SHAPE, ASK_POS  # noqa: E402

K_SHAPE = 3
K_POS = 3
ALL = [(s, p) for s in range(K_SHAPE) for p in range(K_POS)]


def _run(route, n_steps=8000, seed=1):
    g = DialogueGame(K_SHAPE, K_POS, route_by_question=route, seed=seed)
    rng = np.random.default_rng(seed + 50)
    for t in range(n_steps):
        s, p = ALL[int(rng.integers(len(ALL)))]
        q = int(rng.integers(2))
        g.turn(s, p, q)
    return g


def test_responde_certo_as_duas_perguntas():
    """Roteando pela pergunta, o agente responde forma e posição corretamente."""
    g = _run(True)
    assert g.accuracy(ALL, ASK_SHAPE) > 0.9
    assert g.accuracy(ALL, ASK_POS) > 0.9


def test_dialogo_emerge_sem_professor():
    """Sem professor, o entendimento por turno fica bem acima do acaso nas duas perguntas."""
    g = _run(True)
    chance = 0.5 * (1 / K_SHAPE + 1 / K_POS)
    assert g.accuracy_both(ALL) > 2 * chance


def test_cego_a_pergunta_falha_na_posicao():
    """Ablação: o agente CEGO à pergunta acerta a forma mas erra a posição (~acaso).
    Prova que USAR a pergunta é o que permite responder ao que foi perguntado."""
    g = _run(False)
    assert g.accuracy(ALL, ASK_SHAPE) > 0.9            # ainda descreve a forma
    assert g.accuracy(ALL, ASK_POS) < 0.5              # mas não sabe responder a posição


def test_rotear_supera_ser_cego():
    """Usar a pergunta dá acurácia média claramente maior que ser cego a ela."""
    g_route = _run(True)
    g_blind = _run(False)
    assert g_route.accuracy_both(ALL) > g_blind.accuracy_both(ALL) + 0.2


def test_a_pergunta_seleciona_a_resposta():
    """Para a MESMA cena, perguntas diferentes levam a respostas que decodificam atributos
    diferentes (a pergunta seleciona o que descrever)."""
    g = _run(True)
    # escolhe uma cena cujo (forma, posição) tenham valores distintos para o teste ser claro
    s, p = 0, 2
    w_shape = g.answer(s, p, ASK_SHAPE, explore=False)
    w_pos = g.answer(s, p, ASK_POS, explore=False)
    assert g.interpret(w_shape, ASK_SHAPE) == s        # "que forma?" -> a forma certa
    assert g.interpret(w_pos, ASK_POS) == p            # "que posição?" -> a posição certa
