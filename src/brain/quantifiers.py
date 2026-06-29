"""Quantificadores: "todos", "algum", "nenhum" — M36 (semântica de verdade).

Os marcos de linguagem descreviam objetos. Mas a linguagem também faz AFIRMAÇÕES sobre
CONJUNTOS, com QUANTIFICADORES: "TODOS são vermelhos", "ALGUM é vermelho", "NENHUM é
vermelho". Avaliar isso não é descrever — é checar uma frase CONTRA o mundo (semântica de
valor de verdade): percorrer o conjunto e aplicar a lógica do quantificador (∀, ∃, ¬∃).

Cena: vários objetos, cada um com uma cor. O agente recebe (quantificador, cor) e responde
VERDADEIRO/FALSO:
  • todos(c)   = todos os objetos têm a cor c        (∀)
  • algum(c)   = pelo menos um tem a cor c           (∃)
  • nenhum(c)  = nenhum tem a cor c                  (¬∃)

A cor é PERCEBIDA (o agente classifica cada objeto com seu próprio grounding); a lógica do
quantificador é aplicada sobre as percepções.

Pergunta científica (H19): o agente avalia frases QUANTIFICADAS corretamente — aplicando a
lógica de ∀/∃/¬∃ sobre o conjunto percebido — em vez de só nomear?

Honesto: 3 quantificadores, conjuntos pequenos, 1 propriedade (cor), escala de brinquedo. É o
MECANISMO da quantificação, NÃO semântica plena (sem quantificadores aninhados, sem escopo,
sem quantificação sobre relações) — horizonte distante.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np

from .dialogue import DialogueGame, ASK_SHAPE

ALL = 0      # "todos"
SOME = 1     # "algum"
NONE = 2     # "nenhum"
QUANT_NAMES = ["todos", "algum", "nenhum"]


class QuantifierGame:
    """Avalia afirmações quantificadas (todos/algum/nenhum) sobre conjuntos de objetos.

    Reusa o grounding de cor de um `DialogueGame` para PERCEBER a cor de cada objeto; a lógica
    do quantificador é aplicada sobre as cores percebidas.

    `correct_logic=False` é o BASELINE ablado: confunde os quantificadores (aplica sempre a
    lógica de "algum"), então erra "todos"/"nenhum" sistematicamente.
    """

    def __init__(self, k_color: int, correct_logic: bool = True,
                 lr: float = 0.2, temp: float = 0.4, init_noise: float = 0.05, seed: int = 0):
        self.k_color = k_color
        self.correct_logic = correct_logic
        self.dlg = DialogueGame(k_color, k_color, route_by_question=True,
                                lr=lr, temp=temp, init_noise=init_noise, seed=seed)
        self.rng = np.random.default_rng(seed)

    def train_grounding(self, n_steps, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        for _ in range(n_steps):
            c = int(rng.integers(self.k_color))
            self.dlg.turn(c, 0, ASK_SHAPE)

    def perceive(self, color_obs_word):
        """A cor percebida de um objeto, a partir do seu padrão/palavra de cor."""
        return self.dlg.interpret(color_obs_word, ASK_SHAPE)

    # ---------- avaliar a frase quantificada CONTRA a cena ----------
    def evaluate(self, quant, color, scene_colors):
        """`scene_colors` = lista das cores dos objetos da cena (já percebidas).
        Retorna True/False para a afirmação (quant, color) sobre a cena."""
        has = [c == color for c in scene_colors]
        if not self.correct_logic:
            return any(has)                              # baseline: sempre a lógica de "algum"
        if quant == ALL:
            return all(has)
        if quant == SOME:
            return any(has)
        return not any(has)                              # NONE

    # ---------- avaliação: acurácia por quantificador sobre cenas aleatórias ----------
    def eval_accuracy(self, n_scenes=400, set_size=3, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        per_q = {ALL: [], SOME: [], NONE: []}
        for _ in range(n_scenes):
            scene = [int(rng.integers(self.k_color)) for _ in range(set_size)]
            color = int(rng.integers(self.k_color))
            q = int(rng.integers(3))
            # verdade de referência (lógica correta)
            has = [c == color for c in scene]
            truth = all(has) if q == ALL else (any(has) if q == SOME else not any(has))
            pred = self.evaluate(q, color, scene)
            per_q[q].append(int(pred == truth))
        return {QUANT_NAMES[q]: float(np.mean(v)) for q, v in per_q.items()}

    def overall_accuracy(self, **kw):
        accs = self.eval_accuracy(**kw)
        return float(np.mean(list(accs.values())))
