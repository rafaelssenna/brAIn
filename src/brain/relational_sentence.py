"""Frases relacionais e RECURSÃO — M33 (o poder infinito da linguagem).

Os marcos anteriores descreviam UM objeto (forma, cor, posição). Mas a linguagem é
RECURSIVA: uma frase pode conter frases. "A barra vermelha ACIMA do ponto azul" descreve
DOIS objetos e uma RELAÇÃO entre eles — e cada objeto é, ele mesmo, uma sub-descrição. É
isso que dá à linguagem seu poder infinito: estruturas dentro de estruturas.

Este marco dá o passo: uma cena tem dois objetos (cada um = forma × cor) e uma RELAÇÃO
(acima/abaixo). O agente monta/entende uma frase RELACIONAL de tamanho maior, composta de
duas sub-frases de objeto + uma palavra de relação:

    [forma1 cor1]  RELAÇÃO  [forma2 cor2]
    "barra vermelho  acima  ponto azul"

E generaliza: descreve pares de objetos + relações que NUNCA viu, porque entende as PARTES
(os objetos, a relação) e a estrutura que as combina — a composicionalidade recursiva.

Pergunta científica (H16): o agente compõe e entende frases RECURSIVAS (objeto-relação-objeto)
e generaliza para cenas relacionais nunca vistas — descrevendo estruturas dentro de estruturas?

Honesto: recursão de UM nível (objeto dentro de relação), tamanho fixo nessa estrutura, escala
de brinquedo. É o MECANISMO da composição recursiva, NÃO recursão arbitrariamente profunda nem
sintaxe plena — horizonte distante, provável híbrido.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np

from .rich_sentence import RichSentenceLearner


class RelationalSentenceLearner:
    """Aprende frases relacionais objeto-RELAÇÃO-objeto, com sub-frases de objeto recursivas.

    Reusa um `RichSentenceLearner` para a SUB-FRASE de objeto (forma × cor) — a parte
    recursiva: o mesmo aprendiz de objeto é aplicado às duas posições da frase relacional.
    Acrescenta um léxico para a palavra de RELAÇÃO (acima/abaixo). A estrutura da frase é:

        sub-frase(obj1)  +  palavra-relação  +  sub-frase(obj2)

    Aprende tudo só ouvindo, com atenção conjunta (conhece a cena verdadeira)."""

    def __init__(self, k_shape: int, k_color: int, k_rel: int, lr: float = 0.2, seed: int = 0):
        self.k_shape, self.k_color, self.k_rel = k_shape, k_color, k_rel
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        # aprendiz recursivo da sub-frase de objeto (forma × cor) — REUSADO nos dois objetos
        self.obj = RichSentenceLearner([k_shape, k_color], lr=lr, seed=seed)
        self.obj_words = k_shape + k_color           # tamanho do vocabulário de objeto
        # léxico de RELAÇÃO (palavras próprias, após o bloco de objeto)
        rn = lambda *s: 0.01 * self.rng.standard_normal(s)
        self.prod_rel = rn(k_rel, k_rel)             # relação -> palavra de relação
        self.comp_rel = rn(k_rel, k_rel)             # palavra de relação -> relação

    # ---------- ouvir uma frase relacional inteira ----------
    def hear(self, sentence, obj1, rel, obj2):
        """sentence = sub1(2 palavras) + [palavra_rel] + sub2(2 palavras). obj = (forma,cor)."""
        n = self.obj.n_slots                          # 2 palavras por objeto
        sub1 = sentence[:n]
        rel_word = sentence[n] - self.obj_words       # índice local da palavra de relação
        sub2 = sentence[n + 1:]
        self.obj.hear(sub1, obj1)                      # recursão: aprende a sub-frase do obj1
        self.obj.hear(sub2, obj2)                      # ...e do obj2 (mesmo aprendiz)
        self.prod_rel[rel, rel_word] += self.lr
        self.comp_rel[rel_word, rel] += self.lr

    # ---------- produzir / entender ----------
    def say(self, obj1, rel, obj2):
        """Monta a frase relacional inteira: sub(obj1) + relação + sub(obj2)."""
        sub1 = self.obj.say(obj1)
        sub2 = self.obj.say(obj2)
        rel_word = int(np.argmax(self.prod_rel[rel])) + self.obj_words
        return sub1 + [rel_word] + sub2

    def understand(self, sentence):
        """Decompõe a frase relacional em (obj1, relação, obj2)."""
        n = self.obj.n_slots
        sub1 = sentence[:n]
        rel_word = sentence[n] - self.obj_words
        sub2 = sentence[n + 1:]
        obj1 = self.obj.understand(sub1)
        rel = int(np.argmax(self.comp_rel[rel_word]))
        obj2 = self.obj.understand(sub2)
        return obj1, rel, obj2


def make_relational_language(obj_order, k_shape, k_color):
    """Devolve uma função (obj1, rel, obj2) -> frase. `obj_order` é a ordem dos slots DENTRO
    da sub-frase de objeto (ex.: (0,1)=forma,cor). A relação fica sempre no meio.

    Vocabulário: objeto usa [0 .. k_shape+k_color); relação usa [k_shape+k_color .. +k_rel)."""
    obj_words = k_shape + k_color
    obj_offset = [0, k_shape]                          # forma em [0..), cor em [k_shape..)

    def obj_sentence(obj):
        return [obj_offset[a] + obj[a] for a in obj_order]

    def to_sentence(obj1, rel, obj2):
        return obj_sentence(obj1) + [obj_words + rel] + obj_sentence(obj2)
    return to_sentence
