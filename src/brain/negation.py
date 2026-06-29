"""Negação e contraste — M34 (operadores lógicos sobre a referência).

Até aqui o agente descrevia o que VÊ. Mas a linguagem também NEGA e CONTRASTA: "NÃO o
vermelho", "o OUTRO". Isso não é descrição — é um OPERADOR LÓGICO que transforma o conjunto
de referentes: "não X" seleciona o complemento de X. Entender isso é raciocinar sobre a
linguagem, não só nomear.

Cena: dois objetos distintos (cada um = forma × cor, mas aqui usamos UM atributo de contraste
por simplicidade — a cor). O falante aponta um objeto por NEGAÇÃO do outro:

    objetos: [vermelho] e [azul]
    "não o vermelho"  -> o ouvinte escolhe o AZUL (o complemento)
    "o outro" (após mencionar o vermelho) -> também o azul

Pergunta científica (H17): o agente entende a NEGAÇÃO — identificar o objeto pelo que ele NÃO
é (o complemento) — em vez de pelo que ele é? Um agente que ignora o "não" escolhe o errado.

Honesto: negação simples sobre 2 objetos e 1 atributo de contraste, escala de brinquedo. É o
MECANISMO do operador de negação/contraste, NÃO lógica plena (sem quantificadores, sem escopo
aninhado, sem negação de relações) — horizonte distante.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np

from .dialogue import DialogueGame, ASK_SHAPE


class NegationGame:
    """Aponta um objeto por NEGAÇÃO do outro, entre 2 objetos de cores distintas.

    Reusa o grounding de cor de um `DialogueGame` (palavra↔cor). A negação é um operador:
    ao ouvir "não <palavra-cor>", o ouvinte decodifica a cor negada e escolhe, entre os 2
    objetos presentes, o que NÃO tem aquela cor (o complemento).

    `understand_negation=False` é o BASELINE ablado: ignora o "não" e escolhe o objeto que
    CASA com a palavra — exatamente o errado.
    """

    def __init__(self, k_color: int, understand_negation: bool = True,
                 lr: float = 0.2, temp: float = 0.4, init_noise: float = 0.05, seed: int = 0):
        self.k_color = k_color
        self.understand_negation = understand_negation
        # usa o DialogueGame só pelo grounding de cor (tratamos cor como o atributo "shape")
        self.dlg = DialogueGame(k_color, k_color, route_by_question=True,
                                lr=lr, temp=temp, init_noise=init_noise, seed=seed)
        self.rng = np.random.default_rng(seed)

    # ---------- treino do grounding de cor (palavra <-> cor) ----------
    def train_grounding(self, n_steps, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        for _ in range(n_steps):
            c = int(rng.integers(self.k_color))
            self.dlg.turn(c, 0, ASK_SHAPE)              # ASK_SHAPE = pergunta a "cor" (slot 0)

    def color_word(self, color, explore=False):
        """A palavra que nomeia uma cor (produção)."""
        return self.dlg.answer(color, 0, ASK_SHAPE, explore=explore)

    def decode_color(self, word):
        """A cor que uma palavra nomeia (compreensão)."""
        return self.dlg.interpret(word, ASK_SHAPE)

    # ---------- resolver "não <cor>" entre os objetos presentes ----------
    def resolve_negation(self, word, objects):
        """`objects` = lista de cores presentes (2). Ouve "não <word>": escolhe o objeto que
        NÃO tem a cor negada (o complemento). Retorna o índice do objeto escolhido."""
        negated = self.decode_color(word)
        if self.understand_negation:
            cands = [i for i, c in enumerate(objects) if c != negated]   # complemento
        else:
            cands = [i for i, c in enumerate(objects) if c == negated]   # baseline: ignora o "não"
        if not cands:
            return int(self.rng.integers(len(objects)))
        return cands[0]

    # ---------- avaliação ----------
    def eval_negation(self, color_pairs):
        """Para cada par de cores distintas (cA, cB): o falante quer apontar cB dizendo "não cA".
        Acerta se o ouvinte escolhe o objeto de cor cB (o complemento)."""
        ok = 0
        for (cA, cB) in color_pairs:
            objects = [cA, cB]
            word = self.color_word(cA)                  # nomeia a cor NEGADA
            choice = self.resolve_negation(word, objects)
            ok += int(objects[choice] == cB)            # deve escolher o complemento (cB)
        return ok / len(color_pairs)
