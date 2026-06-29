"""Testes do M30 — diálogo de vários turnos com contexto.

Verifica que o agente mantém um OBJETO EM FOCO e RESOLVE A REFERÊNCIA ("e a posição dela?")
ao turno anterior — algo que um agente SEM memória de contexto (turnos isolados) não
consegue, caindo ao acaso.

Roda com:  python -m pytest tests/test_contextual_dialogue.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

np.seterr(all="ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import (ContextualDialogue, ASK_SHAPE_NEW, ASK_POS_NEW,  # noqa: E402
                   ASK_POS_REF, ASK_SHAPE_REF)

K_SHAPE = 3
K_POS = 3
ALL = [(s, p) for s in range(K_SHAPE) for p in range(K_POS)]


def _trained(use_ctx, seed=1, n=8000):
    g = ContextualDialogue(K_SHAPE, K_POS, use_context=use_ctx, seed=seed)
    g.train_grounding(ALL, n, rng_seed=seed + 50)
    return g


def test_com_contexto_resolve_a_referencia():
    """Com memória de foco, o agente resolve 'e a posição dela?' e 'e a forma dela?'."""
    g = _trained(True)
    assert g.eval_reference(ALL, ask_pos_ref=True) > 0.9
    assert g.eval_reference(ALL, ask_pos_ref=False) > 0.9


def test_sem_contexto_falha_na_referencia():
    """Ablação: sem memória de contexto, a referência cai ao acaso (não sabe a quem 'dela'
    se refere)."""
    g = _trained(False)
    assert g.eval_reference(ALL, ask_pos_ref=True) < 0.6     # ~acaso (1/3)
    assert g.eval_reference(ALL, ask_pos_ref=False) < 0.6


def test_contexto_supera_sem_contexto():
    """A memória de contexto dá vantagem clara sobre tratar cada turno isolado."""
    gc = _trained(True)
    gn = _trained(False)
    assert gc.eval_reference(ALL, ask_pos_ref=True) > gn.eval_reference(ALL, ask_pos_ref=True) + 0.3


def test_o_objeto_entra_e_fica_em_foco():
    """Após um turno sobre um objeto novo, ele fica no foco e pode ser referido."""
    g = _trained(True)
    g.reset_focus()
    assert g.focus is None
    g.step(ASK_SHAPE_NEW, scene=(1, 2), learn=False, explore=False)
    assert g.focus == (1, 2)                                 # o objeto entrou em foco
    # o turno referencial responde sobre o foco (posição 2)
    word, reward = g.step(ASK_POS_REF, true_referent=(1, 2), learn=False, explore=False)
    assert reward == 1


def test_dialogo_de_dois_turnos_coerente():
    """Um diálogo completo de 2 turnos: forma e depois 'a posição dela' batem com a cena."""
    g = _trained(True)
    for (s, p) in ALL:
        g.reset_focus()
        w1, _ = g.step(ASK_SHAPE_NEW, scene=(s, p), learn=False, explore=False)
        assert g.dlg.interpret(w1, 0) == s                  # turno 1: a forma certa
        w2, _ = g.step(ASK_POS_REF, true_referent=(s, p), learn=False, explore=False)
        assert g.dlg.interpret(w2, 1) == p                  # turno 2: a posição dela, certa
