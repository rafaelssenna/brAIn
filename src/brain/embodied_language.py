"""Organismo corporificado que fala — M25 (a costura que faltava).

Até aqui o brAIn tinha duas metades separadas do cérebro:
  • um agente com CORPO (M7/M13): navega um mundo, percebe, é curioso, lembra —
    mas é MUDO (não tem linguagem);
  • um agente que FALA (M20→M24): percebe (com spikes), lembra e fala — mas é um
    "cérebro num pote", SEM corpo (os objetos chegam até ele).

Um cérebro humano é UM organismo só: vê, se move, lembra, é curioso E fala, no mesmo
laço. Este marco costura as duas metades. Cada agente é um `LivingAgent` (percepção
spiking/densa + memória + linguagem, M24) montado num CORPO que vive num `RingWorld`
(anel de lugares, cada lugar tem um objeto). A CURIOSIDADE (learning progress, M5)
decide para onde o corpo navega; a percepção aprende o que há lá; a linguagem nomeia.

A diferença crucial para o M20/M21: lá os objetos eram MOSTRADOS ao agente (percepção
passiva). Aqui o agente só vê um objeto se FOR FISICAMENTE até o lugar dele. A
percepção depende da AÇÃO. Pergunta científica (H8): a linguagem ainda emerge quando
ver custa andar? Dois agentes só podem nomear um objeto se ambos foram até ele.

Honesto: escala de brinquedo (poucos lugares, D pequeno). É o MECANISMO do organismo
inteiro num laço, não um cérebro pronto.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np

from .living import LivingAgent
from .curiosity import IntrinsicMotivation


class EmbodiedLanguageAgent:
    """Um `LivingAgent` (percebe+lembra+fala) com CORPO e CURIOSIDADE num anel.

    O agente ocupa uma posição no anel. A cada instante: sente o objeto de onde
    está (se houver), aprende a percebê-lo, a curiosidade observa o progresso ali,
    e o corpo dá um passo rumo ao lugar de maior learning progress. A linguagem é
    a do `LivingAgent` (matrizes símbolo↔conceito), exercida quando dois agentes se
    encontram no mesmo lugar (atenção conjunta).
    """

    def __init__(self, world, n_latent: int, n_symbols: int,
                 spiking: bool = False, sparse_k: int | None = None,
                 eps: float = 0.1, dwell: int = 4, seed: int = 0):
        self.world = world
        self.mind = LivingAgent(world.D, n_latent, n_symbols, seed=seed,
                                spiking=spiking, sparse_k=sparse_k)
        self.mot = IntrinsicMotivation(n_regions=world.n)
        self.eps = eps
        self.dwell_max = dwell
        self.pos = 0
        self.target = None
        self._dwell = 0
        self.rng = np.random.default_rng(seed)

    # --- um instante de vida corporificada: sente onde está, aprende, decide, move ---
    def sense_here(self, rng) -> np.ndarray:
        return self.world.sense(self.pos, rng)

    def perceive_and_learn(self, rng):
        """Sente o objeto do lugar atual e aprende a percebê-lo. Retorna (pos, surpresa)."""
        here = self.pos
        x = self.sense_here(rng)
        err = self.mind.learn_perception(x)        # percepção (spiking/densa) + memória
        self.mot.update(here, err)                 # curiosidade observa o progresso aqui
        return here, err

    def navigate(self):
        """A curiosidade escolhe o alvo (maior learning progress); o corpo dá um passo."""
        n = self.world.n
        if self.target is None or (self.pos == self.target and self._dwell >= self.dwell_max):
            if self.rng.random() < self.eps:
                self.target = int(self.rng.integers(n))            # exploração
            else:
                self.target = int(np.argmax(self.mot.learning_progress()))
            self._dwell = 0
        if self.pos == self.target:
            self._dwell += 1                                       # fica e amostra
        else:
            self.pos = self._step_toward(self.target, n)

    def _step_toward(self, target, n):
        forward = (target - self.pos) % n
        return (self.pos + (1 if forward <= n // 2 else -1)) % n

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


def communication_accuracy(speaker: EmbodiedLanguageAgent, listener: EmbodiedLanguageAgent,
                           clean_objs) -> float:
    """Acurácia de comunicação sobre os objetos limpos (crédito repartido em empates).

    Mede, sem mover: o falante vê o objeto o, diz um símbolo; o ouvinte decodifica e
    aponta um objeto. Acerta se o objeto apontado é o o (repartido se houver ambiguidade).
    """
    lc = [listener.concept_of(clean_objs[j]) for j in range(len(clean_objs))]
    tot = 0.0
    for o in range(len(clean_objs)):
        m = speaker.speak(speaker.concept_of(clean_objs[o]), explore=False)
        cands = [j for j in range(len(clean_objs)) if lc[j] == listener.listen(m)]
        if o in cands:
            tot += 1.0 / len(cands)
    return tot / len(clean_objs)
