"""Testes do mundo rico + RichBrain (infra de escala para treino pesado).

Verifica que o mundo rico gera muitos objetos distinguíveis, que o RichBrain aprende a
nomear os atributos (ao menos os fáceis) e que salva/carrega (memória de longo prazo).

Roda com:  python -m pytest tests/test_rich_world.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

np.seterr(all="ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import RichBrain  # noqa: E402
from brain.rich_world import (D, n_objects, all_objects, render, describe,  # noqa: E402
                              N_WORDS, N_ATTR, ATTR_VALUES)


def test_mundo_tem_muitos_objetos():
    """O mundo rico é grande: dezenas a centenas de objetos distintos (escala para treino)."""
    assert n_objects() == np.prod([len(v) for v in ATTR_VALUES])
    assert n_objects() >= 100                          # bem maior que os 6 do mundo antigo
    assert len(all_objects()) == n_objects()


def test_render_produz_imagem_normalizada():
    """Cada objeto vira uma imagem colorida achatada (D,), com norma unitária."""
    rng = np.random.default_rng(0)
    v = render((0, 0, 0, 0), rng)
    assert v.shape == (D,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_objetos_diferentes_geram_imagens_diferentes():
    """Objetos com atributos distintos produzem imagens distinguíveis (sem ruído)."""
    a = render((0, 0, 0, 0))
    b = render((1, 1, 2, 3))
    assert np.linalg.norm(a - b) > 0.3                 # bem diferentes


def test_describe_da_uma_palavra_por_atributo():
    """A descrição de um objeto tem uma palavra por atributo, dos blocos certos."""
    words = describe((1, 2, 0, 3))
    assert len(words) == N_ATTR
    assert all(0 <= w < N_WORDS for w in words)
    assert len(set(words)) == N_ATTR                   # palavras de atributos distintos


def test_richbrain_aprende_a_nomear_atributos():
    """Treinando, o RichBrain aprende a nomear (ao menos a posição, que é fácil) acima do acaso."""
    brain = RichBrain(n_latent=24, seed=0)
    objs = all_objects()
    rng = np.random.default_rng(0)
    for _ in range(8000):
        brain.learn(objs[int(rng.integers(len(objs)))], rng)
    test = [objs[i] for i in rng.choice(len(objs), 40, replace=False)]
    acc = brain.naming_accuracy(test)
    # posição é o atributo fácil (quadrantes bem separados) -> deve ir bem acima do acaso
    assert acc["posicao"] > 0.6                        # acaso = 1/4
    # e a média geral fica acima do acaso médio
    chance = np.mean([1 / len(v) for v in ATTR_VALUES])
    assert np.mean(list(acc.values())) > chance + 0.1


def test_richbrain_salva_e_carrega():
    """O cérebro rico tem memória de longo prazo: salvar e carregar preserva o aprendido."""
    brain = RichBrain(n_latent=24, seed=0)
    objs = all_objects()
    rng = np.random.default_rng(0)
    for _ in range(3000):
        brain.learn(objs[int(rng.integers(len(objs)))], rng)
    test = [objs[i] for i in rng.choice(len(objs), 30, replace=False)]
    before = brain.naming_accuracy(test)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "brain_rich")
        brain.save(path)
        loaded = RichBrain.load(path)
    after = loaded.naming_accuracy(test)
    assert after == before                             # acorda igual
