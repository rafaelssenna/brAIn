"""Diálogo com TURNOS: pergunta → resposta ancorada — M29 (do conversar ao dialogar).

O M28 já fazia dois agentes conversarem com frases (mão única: um descreve, o outro
adivinha). Dialogar de verdade tem TROCA: um PERGUNTA, o outro RESPONDE sobre o que percebe.
Este marco dá esse passo. Dois agentes olham a mesma cena (uma FORMA numa POSIÇÃO). O
PERGUNTADOR faz uma de duas perguntas — "que forma?" ou "que posição?" — e o RESPONDEDOR,
percebendo a cena, diz SÓ a palavra certa para aquela pergunta (não a frase inteira).

A capacidade NOVA: a pergunta SELECIONA o que descrever. O respondedor tem que aprender a
ROTEAR — responder forma quando perguntam forma, posição quando perguntam posição. É o
começo de raciocinar sobre a linguagem: responder ao que foi perguntado, não despejar tudo.

Sem professor dando a resposta: a recompensa vem de o perguntador (que conhece o atributo
verdadeiro que perguntou) confirmar se a resposta bate. Ambos aprendem (Hebbiano de 3
fatores + atenção conjunta). O vocabulário de respostas é ancorado (cada palavra ↔ um valor
de atributo percebido).

Pergunta científica (H12): um agente aprende a RESPONDER À PERGUNTA — usar o tipo de pergunta
para escolher QUAL parte da cena descrever — só dialogando, sem ninguém ensinar?

Honesto: escala de brinquedo (2 tipos de pergunta, poucos atributos, resposta de 1 palavra).
É o MECANISMO do turno pergunta→resposta com seleção, NÃO diálogo pleno (sem sintaxe rica,
sem múltiplos turnos encadeados, sem semântica profunda) — horizonte distante, provável híbrido.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np

ASK_SHAPE = 0
ASK_POS = 1


class DialogueGame:
    """Perguntador + respondedor que dialogam sobre cenas (forma × posição) por turnos.

    O respondedor mantém DOIS léxicos de resposta — um para perguntas de forma, outro para
    perguntas de posição — e aprende, por interação, a usar o atributo certo conforme a
    pergunta. O perguntador conhece o atributo que perguntou (atenção conjunta) e recompensa.

    `route_by_question=False` é o BASELINE ablado: o respondedor é CEGO à pergunta — não a
    enxerga, então responde sempre sobre UM atributo fixo (a forma). Quando perguntam a
    posição, ele erra (responde forma). Acerta ~só metade (as perguntas de forma). Prova que
    USAR a pergunta para escolher o que dizer é o que faz o diálogo funcionar.
    """

    def __init__(self, k_shape: int, k_pos: int, n_words: int | None = None,
                 route_by_question: bool = True, lr: float = 0.2, temp: float = 0.4,
                 init_noise: float = 0.05, seed: int = 0):
        self.k_shape, self.k_pos = k_shape, k_pos
        self.W = n_words or (k_shape + k_pos + 1)        # vocabulário de respostas
        self.route = route_by_question
        self.lr = lr
        self.temp = temp
        self.rng = np.random.default_rng(seed)
        rn = lambda *s: init_noise * self.rng.standard_normal(s)
        if route_by_question:
            # respondedor: por TIPO DE PERGUNTA, atributo -> palavra (produção)
            self.A_shape = rn(k_shape, self.W)           # pergunta forma:  forma -> palavra
            self.A_pos = rn(k_pos, self.W)               # pergunta posição: posição -> palavra
            # perguntador: por tipo de pergunta, palavra -> valor do atributo (compreensão)
            self.U_shape = rn(self.W, k_shape)
            self.U_pos = rn(self.W, k_pos)
        else:                                            # baseline CEGO: só descreve a forma
            self.A_one = rn(k_shape, self.W)             # cena -> palavra (sempre a forma)
            self.U_one = rn(self.W, k_shape)

    def _sm(self, v):
        z = np.exp((v - v.max()) / self.temp); return z / z.sum()

    def _pick(self, prefs, explore):
        return int(self.rng.choice(len(prefs), p=self._sm(prefs))) if explore else int(np.argmax(prefs))

    # ---------- responder: vê a cena + ouve a pergunta -> diz a palavra ----------
    def answer(self, shape, pos, question, explore=True):
        if self.route:
            if question == ASK_SHAPE:
                return self._pick(self.A_shape[shape], explore)
            return self._pick(self.A_pos[pos], explore)
        else:
            return self._pick(self.A_one[shape], explore)      # CEGO à pergunta: só descreve a forma

    # ---------- perguntador: ouve a resposta -> interpreta o valor ----------
    def interpret(self, word, question):
        if self.route:
            if question == ASK_SHAPE:
                return int(np.argmax(self.U_shape[word]))
            return int(np.argmax(self.U_pos[word]))
        else:
            # baseline cego: a resposta é sempre sobre a forma. Numa pergunta de posição, a
            # interpretação (um valor de forma) quase nunca bate com a posição verdadeira.
            return int(np.argmax(self.U_one[word]))

    # ---------- um turno de diálogo (sem professor) ----------
    def turn(self, shape, pos, question, learn=True, explore=True):
        """Pergunta sobre a cena; respondedor responde; perguntador interpreta. Retorna 1 se
        a interpretação bate com o atributo verdadeiro perguntado."""
        truth = shape if question == ASK_SHAPE else pos
        word = self.answer(shape, pos, question, explore=explore)
        guess = self.interpret(word, question)
        reward = int(guess == truth)
        if learn:
            d = self.lr if reward else -self.lr
            if self.route:
                if question == ASK_SHAPE:
                    self.A_shape[shape, word] += d
                    self.U_shape[word, shape] += self.lr      # atenção conjunta (sabe o alvo)
                else:
                    self.A_pos[pos, word] += d
                    self.U_pos[word, pos] += self.lr
            else:
                # baseline cego: só aprende a descrever a forma (não vê a pergunta).
                self.A_one[shape, word] += d
                self.U_one[word, shape] += self.lr
        return reward

    # ---------- avaliação ----------
    def accuracy(self, scenes, question):
        """Acurácia determinística de um tipo de pergunta sobre as cenas dadas."""
        ok = 0
        for (s, p) in scenes:
            truth = s if question == ASK_SHAPE else p
            word = self.answer(s, p, question, explore=False)
            ok += int(self.interpret(word, question) == truth)
        return ok / len(scenes)

    def accuracy_both(self, scenes):
        """Acurácia média nas duas perguntas (forma e posição)."""
        return 0.5 * (self.accuracy(scenes, ASK_SHAPE) + self.accuracy(scenes, ASK_POS))
