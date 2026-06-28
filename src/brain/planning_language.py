"""Organismo que PLANEJA rotas num mundo 2D e fala — M26 (a unificação 2D).

O M25 deu corpo ao organismo que fala, mas num anel simples (`RingWorld`): andar era
trivial, sem obstáculos. Um cérebro de verdade não só se move — ele PLANEJA: imagina o
caminho, contorna paredes, alcança o que quer. Esse maquinário já existia no projeto
(M11/M13: `GridWorld` com paredes + `WorldModel` aprendido + `plan()` por busca na
imaginação), mas vivia no agente MUDO. Aqui ele se junta à linguagem (M16/M21/M25).

Cada agente é um `LivingAgent` (percepção spiking/densa + memória + linguagem, M24/M25)
montado num CORPO que vive num `GridWorld` 2D com PAREDES. Ele:
  • aprende o MODELO DE TRANSIÇÃO vivendo (nasce sem saber o que suas ações fazem);
  • a CURIOSIDADE (learning progress, M5) escolhe a célula-meta (o objeto a estudar);
  • o PLANEJAMENTO (M11) traça a rota até lá — contornando paredes que o reativo não vence;
  • PERCEBE o objeto da célula, LEMBRA e FALA sobre ele.

A linguagem é aprendida por ATENÇÃO CONJUNTA, como uma criança (M21): quando o agente
CHEGA a uma célula de objeto, um "professor" diz a palavra certa, e ele aprende a NOMEAR
(vê → diz a palavra) e a APONTAR (ouve → identifica). Só aprende a palavra de um objeto se
CHEGAR à célula dele — e alguns objetos ficam ATRÁS de uma parede: só quem PLANEJA chega lá.

Pergunta científica (H9): o planejamento permite ao organismo NOMEAR o que está atrás da
parede — algo que o reativo (sem imaginar o desvio) nunca alcança, nunca percebe nem nomeia?

Honesto: escala de brinquedo (grid pequeno, poucos objetos). É o organismo inteiro num
laço 2D (perceber+planejar+mover+lembrar+curiosidade+falar), não um cérebro pronto.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np

from .living import LivingAgent
from .curiosity import IntrinsicMotivation
from .planning import WorldModel, plan, N_ACTIONS


class PlanningLanguageAgent:
    """Um `LivingAgent` (percebe+lembra+fala) com CORPO 2D, MODELO DE MUNDO e PLANEJAMENTO.

    `grid` é um `GridWorld` (transições/paredes). `content_cells` é a lista ordenada de
    células com objeto (o índice na lista é o "conceito/palavra" daquele objeto). A
    curiosidade rastreia o learning progress por célula de conteúdo e escolhe a meta;
    `use_planning=False` cai num agente reativo (anda na direção da meta, trava na parede).
    """

    def __init__(self, grid, content_cells, n_obs: int, n_latent: int, n_symbols: int,
                 start, use_planning: bool = True, spiking: bool = False,
                 sparse_k: int | None = None, eps: float = 0.15, seed: int = 0):
        self.grid = grid
        self.content_cells = list(content_cells)
        self.idx = {c: i for i, c in enumerate(self.content_cells)}
        self.mind = LivingAgent(n_obs, n_latent, n_symbols, seed=seed,
                                spiking=spiking, sparse_k=sparse_k)
        self.tm = WorldModel()                                      # modelo de transição (M11)
        self.mot = IntrinsicMotivation(n_regions=len(self.content_cells))   # curiosidade (M5)
        self.use_planning = use_planning
        self.eps = eps
        self.pos = start
        self.goal = None
        self.rng = np.random.default_rng(seed)

    # --- percepção: aprende o objeto da célula onde está (se houver) ---
    def perceive_and_learn(self, obs_for, rng):
        """`obs_for(cell, rng)` devolve a observação sensorial da célula.
        Retorna (pos, surpresa, observação)."""
        here = self.pos
        x = obs_for(here, rng)
        err = self.mind.learn_perception(x)
        if here in self.idx:
            self.mot.update(self.idx[here], err)                   # curiosidade observa o progresso
        return here, err, x

    def learn_word_here(self, observation):
        """Atenção conjunta (M21): se estou numa célula de objeto, o professor diz a
        palavra certa e eu aprendo a NOMEAR e a APONTAR aquele objeto, a partir do que
        estou vendo (`observation`). Só aprendo a palavra de um objeto se eu tiver
        CHEGADO até a célula dele."""
        if self.pos not in self.idx:
            return
        word = self.idx[self.pos]                                  # a palavra daquele objeto
        c = self.mind.concept(observation)                        # meu conceito do que vejo aqui
        self.mind.learn_listen(word, c)                            # ouço a palavra -> meu conceito
        self.mind.reinforce_speak(c, word, True)                   # meu conceito -> aquela palavra

    # --- decisão: curiosidade escolhe a meta; planejamento traça a rota ---
    def _choose_goal(self):
        if self.rng.random() < self.eps:
            gi = int(self.rng.integers(len(self.content_cells)))   # exploração
        else:
            gi = int(np.argmax(self.mot.learning_progress()))
        self.goal = self.content_cells[gi]

    def navigate(self):
        """Um passo rumo à meta. Planejador imagina a rota; reativo anda reto e trava."""
        if self.goal is None or self.pos == self.goal:
            self._choose_goal()
        if self.use_planning:
            path = plan(self.tm, self.pos, self.goal)              # PENSA a rota (M11)
            a = path[0] if path else int(self.rng.integers(N_ACTIONS))
        else:
            a = self._reactive_action()
        sp = self.grid.step(self.pos, a)
        self.tm.observe(self.pos, a, sp)                           # aprende a transição vivendo
        if sp == self.pos:                                        # travou: cutuca e aprende
            a2 = int(self.rng.integers(N_ACTIONS))
            sp = self.grid.step(self.pos, a2)
            self.tm.observe(self.pos, a2, sp)
        self.pos = sp

    def _reactive_action(self):
        dr = self.goal[0] - self.pos[0]; dc = self.goal[1] - self.pos[1]
        if abs(dr) >= abs(dc):
            return 1 if dr > 0 else 0                              # 0=cima,1=baixo
        return 3 if dc > 0 else 2                                  # 2=esq,3=dir

    # --- atalhos de linguagem/percepção (delegam ao LivingAgent) ---
    def concept_of(self, x) -> int:
        return self.mind.concept(x)

    def speak(self, concept: int, explore: bool = True) -> int:
        return self.mind.speak(concept, explore=explore)

    def listen(self, symbol: int) -> int:
        return self.mind.listen(symbol)

    def reinforce_speak(self, concept: int, symbol: int, reward: bool) -> None:
        self.mind.reinforce_speak(concept, symbol, reward)

    def learn_listen(self, symbol: int, true_concept: int) -> None:
        self.mind.learn_listen(symbol, true_concept)

    def surprise_on(self, x) -> float:
        return self.mind.pc.prediction_error(x)


def communication_accuracy(speaker: PlanningLanguageAgent, listener: PlanningLanguageAgent,
                           clean_objs) -> float:
    """Acurácia de comunicação sobre os objetos limpos (crédito repartido em empates)."""
    lc = [listener.concept_of(clean_objs[j]) for j in range(len(clean_objs))]
    tot = 0.0
    for o in range(len(clean_objs)):
        m = speaker.speak(speaker.concept_of(clean_objs[o]), explore=False)
        cands = [j for j in range(len(clean_objs)) if lc[j] == listener.listen(m)]
        if o in cands:
            tot += 1.0 / len(cands)
    return tot / len(clean_objs)
