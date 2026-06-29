"""Correferência: resolver 'dela' entre VÁRIOS objetos — M31 (a ambiguidade do diálogo).

O M30 manteve UM objeto em foco, então "dela" era trivial — só havia um a que se referir.
Mas uma conversa real fala de VÁRIOS objetos, e "dela" fica AMBÍGUO: qual delas? O cérebro
resolve isso com pistas — a mais recente/saliente, ou a que combina com uma descrição. Este
marco dá esse passo: a cena tem vários objetos, e o agente precisa escolher o referente CERTO.

    cena: [barraH no topo] e [barraV na base]
    P: "que forma a primeira?"   R: "barraH"   (entra em foco)
    P: "e a posição DELA?"       R: "topo"     (resolve: 'dela' = a barraH, recém-mencionada)

Duas formas de resolver a referência:
  • por RECÊNCIA/saliência: "dela" = o objeto mencionado mais recentemente (o default humano);
  • por DESCRIÇÃO: "a barraV, onde está?" — escolhe pelo atributo dito (desambigua explícito).

Pergunta científica (H14): com vários candidatos, o agente resolve 'dela' escolhendo o
referente certo (pela recência ou pela descrição) — algo que escolher ao acaso entre os
candidatos não consegue?

Honesto: escala de brinquedo (2-3 objetos, recência simples, descrição por 1 atributo). É o
MECANISMO da correferência, NÃO resolução plena (sem pronomes de gênero/número, sem cadeias
longas, sem inferência pragmática) — horizonte distante, provável híbrido.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np

from .dialogue import DialogueGame, ASK_SHAPE, ASK_POS


class CoreferenceDialogue:
    """Diálogo sobre cenas com VÁRIOS objetos, resolvendo 'dela' entre os candidatos.

    Mantém uma PILHA DE FOCO (objetos mencionados, do mais antigo ao mais recente). Numa
    pergunta referencial sem descrição, resolve pela RECÊNCIA (o topo da pilha). Com uma
    descrição (um valor de atributo), resolve pelo objeto que CASA com ela.

    `strategy='recency'` (resolve pelo mais recente) vs `'random'` (baseline: escolhe um
    candidato ao acaso — erra quando há mais de um).
    """

    def __init__(self, k_shape: int, k_pos: int, n_words: int | None = None,
                 strategy: str = "recency", lr: float = 0.2, temp: float = 0.4,
                 init_noise: float = 0.05, seed: int = 0):
        self.k_shape, self.k_pos = k_shape, k_pos
        self.strategy = strategy
        self.dlg = DialogueGame(k_shape, k_pos, n_words=n_words, route_by_question=True,
                                lr=lr, temp=temp, init_noise=init_noise, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.focus_stack = []          # objetos em foco, [..., mais_recente]

    # ---------- gestão de foco ----------
    def reset(self):
        self.focus_stack = []

    def mention(self, obj):
        """Um objeto (shape,pos) é mencionado: entra (ou sobe) no topo da pilha de foco."""
        self.focus_stack = [o for o in self.focus_stack if o != obj] + [obj]

    # ---------- resolver a referência entre os candidatos ----------
    def resolve(self, describe=None):
        """Escolhe o referente de 'dela' entre os objetos em foco.
          • describe=('shape', v) ou ('pos', v): escolhe o objeto cujo atributo casa (desambig.);
          • sem describe: pela estratégia ('recency' = topo da pilha; 'random' = ao acaso)."""
        if not self.focus_stack:
            return None
        if describe is not None:
            attr, val = describe
            cands = [o for o in self.focus_stack
                     if (o[0] if attr == "shape" else o[1]) == val]
            if cands:
                return cands[-1]                       # o mais recente que casa
        if self.strategy == "recency":
            return self.focus_stack[-1]                # mais recente = mais saliente
        return self.focus_stack[int(self.rng.integers(len(self.focus_stack)))]   # baseline

    # ---------- treino do grounding (forma/posição), como no M29 ----------
    def train_grounding(self, scenes, n_steps, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        for _ in range(n_steps):
            s, p = scenes[int(rng.integers(len(scenes)))]
            q = int(rng.integers(2))
            self.dlg.turn(s, p, q)

    # ---------- responder uma pergunta sobre o referente resolvido ----------
    def answer_ref(self, ask="pos", describe=None):
        """Resolve 'dela' e responde o atributo pedido. Retorna (objeto_resolvido, valor_dito)."""
        ref = self.resolve(describe=describe)
        if ref is None:
            return None, None
        q = ASK_POS if ask == "pos" else ASK_SHAPE
        word = self.dlg.answer(ref[0], ref[1], q, explore=False)
        return ref, self.dlg.interpret(word, q)

    # ---------- avaliação ----------
    def eval_recency(self, objs_pairs, ask="pos"):
        """Para cada par de objetos (o1, o2): menciona o1, depois o2; pergunta 'dela' (deve
        resolver o2, o mais recente). Fração resolvida correta (objeto E valor certos)."""
        ok = 0
        for (o1, o2) in objs_pairs:
            self.reset(); self.mention(o1); self.mention(o2)
            ref, val = self.answer_ref(ask=ask)
            true = o2                                   # recência: o último mencionado
            true_val = true[1] if ask == "pos" else true[0]
            ok += int(ref == true and val == true_val)
        return ok / len(objs_pairs)

    def eval_described(self, triples, ask="pos"):
        """Para cada (o1, o2, descrição): menciona ambos; pergunta 'a <descrição>, ...' — deve
        escolher o objeto que casa com a descrição, não o mais recente. Fração correta."""
        ok = 0
        for (o1, o2, describe) in triples:
            self.reset(); self.mention(o1); self.mention(o2)
            ref, val = self.answer_ref(ask=ask, describe=describe)
            attr, dv = describe
            target = next((o for o in (o2, o1) if (o[0] if attr == "shape" else o[1]) == dv), None)
            true_val = (target[1] if ask == "pos" else target[0]) if target else None
            ok += int(ref == target and val == true_val)
        return ok / len(triples)
