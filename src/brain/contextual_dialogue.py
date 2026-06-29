"""Diálogo de VÁRIOS TURNOS com CONTEXTO — M30 (a conversa coerente no tempo).

O M29 fez um turno isolado (pergunta→resposta). Mas uma conversa de verdade tem MEMÓRIA:
o que foi dito antes importa, e a gente se refere a isso ("...e a posição DELA?"). Este
marco dá esse passo. Uma conversa tem vários turnos sobre o mesmo objeto:

    Turno 1:  "que forma?"          -> "barraH"     (o objeto entra em FOCO)
    Turno 2:  "e a posição DELA?"   -> "topo"       (referência ao foco — sem remostrar nada)

A capacidade NOVA: manter um OBJETO EM FOCO na memória e RESOLVER A REFERÊNCIA — responder
sobre "ela" usando o que ficou em foco no turno anterior, sem o objeto ser reapresentado.
É o que liga linguagem (M29) à memória (M8) e torna a conversa COERENTE no tempo.

Sem professor: o perguntador conhece o objeto em foco (atenção conjunta) e confirma. O
respondedor reusa os léxicos ancorados do diálogo (forma/posição) — o que é NOVO é o
rastreamento de contexto.

Pergunta científica (H13): um agente sustenta um diálogo de vários turnos, resolvendo uma
referência ("dela") ao objeto que ficou em foco — algo que um agente SEM memória de
contexto (que trata cada turno isolado) não consegue?

Honesto: escala de brinquedo (1 objeto em foco por vez, referência simples, poucos
atributos). É o MECANISMO do contexto de diálogo, NÃO conversa plena (sem múltiplos
referentes, sem correferência ambígua, sem pragmática rica) — horizonte distante.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np

from .dialogue import DialogueGame, ASK_SHAPE, ASK_POS

# tipos de turno
ASK_SHAPE_NEW = 0        # "que forma?" (objeto novo, mostrado) — entra em foco
ASK_POS_NEW = 1          # "que posição?" (objeto novo, mostrado) — entra em foco
ASK_POS_REF = 2          # "e a posição DELA?" (referência ao objeto em foco; nada mostrado)
ASK_SHAPE_REF = 3        # "e a forma DELA?" (referência ao objeto em foco)


class ContextualDialogue:
    """Diálogo de vários turnos com memória de FOCO. Estende o jogo do M29 (mesmos léxicos
    ancorados de forma/posição) com um rastreador de contexto: o último objeto sobre o qual
    se falou fica EM FOCO e pode ser referido ("dela") nos turnos seguintes.

    `use_context=False` é o BASELINE ablado: não há memória de foco. Num turno referencial,
    o respondedor não tem objeto a que se referir e precisa CHUTAR — então erra a referência.
    """

    def __init__(self, k_shape: int, k_pos: int, n_words: int | None = None,
                 use_context: bool = True, lr: float = 0.2, temp: float = 0.4,
                 init_noise: float = 0.05, seed: int = 0):
        self.k_shape, self.k_pos = k_shape, k_pos
        self.use_context = use_context
        # reusa o diálogo de turno único (M29) para o grounding forma/posição
        self.dlg = DialogueGame(k_shape, k_pos, n_words=n_words, route_by_question=True,
                                lr=lr, temp=temp, init_noise=init_noise, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.focus = None        # (shape, pos) do objeto em foco, ou None

    # ---------- turno: pode mostrar um objeto novo OU referir o foco ----------
    def step(self, turn_type, scene=None, true_referent=None, learn=True, explore=True):
        """Um turno do diálogo.
          • ASK_SHAPE_NEW / ASK_POS_NEW: objeto NOVO em `scene=(shape,pos)`; entra em foco.
          • ASK_POS_REF / ASK_SHAPE_REF: refere o objeto EM FOCO (scene é ignorado). O
            `true_referent=(shape,pos)` é o que o PERGUNTADOR tem em mente (para avaliar a
            referência honestamente — independente de o respondedor ter mantido o foco).
        Retorna (palavra_resposta, acerto)."""
        if turn_type in (ASK_SHAPE_NEW, ASK_POS_NEW):
            shape, pos = scene
            if self.use_context:
                self.focus = (shape, pos)               # o objeto entra em foco
            q = ASK_SHAPE if turn_type == ASK_SHAPE_NEW else ASK_POS
            reward = self.dlg.turn(shape, pos, q, learn=learn, explore=explore)
            word = self.dlg.answer(shape, pos, q, explore=False)
            return word, reward

        # turno REFERENCIAL: o respondedor usa o objeto que ELE tem em foco (se tiver).
        if self.use_context and self.focus is not None:
            shape, pos = self.focus                      # lembra o referente
        else:
            # baseline sem contexto (ou foco perdido): não sabe a quem "dela" se refere -> chuta
            shape = int(self.rng.integers(self.k_shape))
            pos = int(self.rng.integers(self.k_pos))
        q = ASK_POS if turn_type == ASK_POS_REF else ASK_SHAPE
        word = self.dlg.answer(shape, pos, q, explore=explore)
        guess = self.dlg.interpret(word, q)
        # avaliação SEMPRE contra o referente verdadeiro do perguntador (não o que o agente
        # inventou). Sem contexto, o chute quase nunca bate -> a referência falha de verdade.
        ref = true_referent if true_referent is not None else self.focus
        if ref is None:
            ref = (shape, pos)
        true_val = ref[1] if turn_type == ASK_POS_REF else ref[0]
        reward = int(guess == true_val)
        return word, reward

    def reset_focus(self):
        self.focus = None

    # ---------- treino do grounding subjacente (forma/posição), como no M29 ----------
    def train_grounding(self, scenes, n_steps, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        for _ in range(n_steps):
            s, p = scenes[int(rng.integers(len(scenes)))]
            q = int(rng.integers(2))
            self.dlg.turn(s, p, q)

    # ---------- avaliar a resolução de referência num diálogo de 2 turnos ----------
    def eval_reference(self, scenes, ask_pos_ref=True):
        """Para cada cena: turno 1 fixa o foco (pergunta a forma), turno 2 pergunta a
        referência ('e a posição/forma dela?'). Fração de referências resolvidas certo."""
        ok = 0
        for (s, p) in scenes:
            self.reset_focus()
            # turno 1: objeto novo entra em foco
            self.step(ASK_SHAPE_NEW, scene=(s, p), learn=False, explore=False)
            # turno 2: referência ao foco — avaliada contra o referente VERDADEIRO (s,p)
            tt = ASK_POS_REF if ask_pos_ref else ASK_SHAPE_REF
            _, reward = self.step(tt, true_referent=(s, p), learn=False, explore=False)
            ok += reward
        return ok / len(scenes)
