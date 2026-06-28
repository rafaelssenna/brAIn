"""Organismo cognitivo vivo — M20 (a costura da cognição).

Junta num ÚNICO agente, num laço online, o que antes vivia em scripts separados:
  • PERCEPÇÃO / conceitos (predictive coder, M4/M9): aprende a categorizar o mundo;
  • MEMÓRIA (replay, M8): reensaia experiências para consolidar os conceitos;
  • LINGUAGEM (M16/M19): fala sobre o que percebe, usando seus próprios conceitos.

A diferença para o M19: lá os conceitos eram aprendidos ANTES, offline. Aqui o
agente aprende a PERCEBER e a FALAR ao mesmo tempo, vivendo. Isso levanta uma
pergunta real: dá para ancorar a linguagem em conceitos que ainda estão se
formando? (Ver experiments/m20_living_mind.py.)
"""

from __future__ import annotations

import numpy as np

from .sparse_predictive import SparsePredictiveCoder
from .spiking_predictive import SpikingPerceptionCoder
from .memory import ReplayBuffer


class LivingAgent:
    """Um organismo que percebe, lembra e fala — tudo aprendido online.

    O substrato de percepção pode ser:
      • DENSO (padrão, idêntico ao M4) — `sparse_k=None, l1=0, spiking=False`;
      • ESPARSO (M22) — passe `sparse_k` (k-WTA) e/ou `l1` para código cortical;
      • SPIKING (M24) — passe `spiking=True` para PERCEBER com neurônios LIF reais
        que disparam (substrato do M10 dentro do organismo). É o passo mais "cérebro
        de verdade": a percepção deixa de ser uma taxa contínua e passa a emergir de
        spikes. A esparsidade aí é EMERGENTE (corrente sublimiar => não dispara).

    Com `sparse_k=None, l1=0, spiking=False` o comportamento é exatamente o do
    M20/M21 (sem regressão). `spiking=True` ignora `sparse_k`/`l1` (substrato distinto).
    """

    def __init__(self, n_obs: int, n_latent: int, n_symbols: int,
                 lr_lang: float = 0.15, temp: float = 0.4,
                 replay_every: int = 10, replay_batch: int = 8, seed: int = 0,
                 sparse_k: int | None = None, l1: float = 0.0,
                 spiking: bool = False):
        if spiking:
            self.pc = SpikingPerceptionCoder(n_obs=n_obs, n_latent=n_latent,
                                             eta_w=0.05, seed=seed)
        else:
            self.pc = SparsePredictiveCoder(n_obs=n_obs, n_latent=n_latent, eta_w=0.05,
                                            seed=seed, k_active=sparse_k, l1=l1)
        self.spiking = spiking
        self.buffer = ReplayBuffer(300, n_obs, seed=seed)
        self.nL = n_latent
        self.M = n_symbols
        self.lr = lr_lang
        self.temp = temp
        self.replay_every = replay_every
        self.replay_batch = replay_batch
        self.rng = np.random.default_rng(seed)
        # linguagem: meu-conceito -> símbolo (falar); símbolo -> meu-conceito (ouvir)
        self.S = 0.05 * self.rng.standard_normal((n_latent, n_symbols))
        self.R = 0.05 * self.rng.standard_normal((n_symbols, n_latent))
        self._t = 0

    # --- percepção (o que estou vendo, segundo o que aprendi até agora) ---
    def concept(self, obj) -> int:
        return int(np.argmax(self.pc.infer(obj)))

    def learn_perception(self, obj) -> float:
        _, err = self.pc.learn(obj)
        self.buffer.add(obj)
        self._t += 1
        if self.replay_every and self._t % self.replay_every == 0 and self.buffer.size:
            for xr in self.buffer.sample(self.replay_batch):
                self.pc.learn(xr)                      # memória: reensaio (M8)
        return err                                     # surpresa (alimenta a curiosidade)

    # --- linguagem ---
    def _sm(self, v):
        z = np.exp((v - v.max()) / self.temp); return z / z.sum()

    def speak(self, concept: int, explore: bool = True) -> int:
        if explore:
            return int(self.rng.choice(self.M, p=self._sm(self.S[concept])))
        return int(np.argmax(self.S[concept]))

    def listen(self, symbol: int) -> int:
        return int(np.argmax(self.R[symbol]))

    def reinforce_speak(self, concept: int, symbol: int, reward: bool) -> None:
        self.S[concept, symbol] += self.lr if reward else -self.lr

    def learn_listen(self, symbol: int, true_concept: int) -> None:
        self.R[symbol, true_concept] += self.lr        # atenção conjunta (cross-situacional)

    # --- medida de percepção: quantos objetos o agente já distingue ---
    def discriminability(self, objects) -> float:
        cs = [self.concept(o) for o in objects]
        return len(set(cs)) / len(objects)
