"""Sinapses plásticas com STDP — M2.

Implementação do zero (NumPy) de conexões sinápticas que APRENDEM por
Spike-Timing-Dependent Plasticity. Ver docs/04-fundacoes-matematicas.md, 2.2.

Regra STDP (versão online, baseada em traços — biologicamente local):

    Cada neurônio pré mantém um traço a_pre que decai com tau_pre.
    Cada neurônio pós mantém um traço a_post que decai com tau_post.

    Quando o PRÉ dispara:   w += a_post   (a_post <= 0  => depressão)
                            a_pre += A_plus
    Quando o PÓS dispara:   w += a_pre    (a_pre  >= 0  => potenciação)
                            a_post += A_minus   (A_minus < 0)

Intuição da causalidade:
  - pré dispara ANTES do pós  => a_pre ainda alto quando o pós dispara
    => w sobe (a sinapse "ajudou a causar" o disparo: é reforçada).
  - pré dispara DEPOIS do pós => a_post (negativo) ainda ativo no disparo do
    pré => w desce (não ajudou: é enfraquecida).

A transmissão é via corrente sináptica com decaimento exponencial
(synapse current-based): cada disparo pré injeta `w` (nA) na corrente do pós,
que decai com tau_syn. Unidades como em neuron.py (ms, mV, MΩ, nA).
"""

from __future__ import annotations

import numpy as np


class STDPConnection:
    """Conexão de n_pre -> n_post neurônios, com pesos plásticos por STDP.

    `w` tem shape (n_pre, n_post). Se `plastic=False`, os pesos ficam fixos
    (útil para a sinapse "incondicionada" do experimento pavloviano).
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        w_init: float | np.ndarray = 0.5,
        w_max: float = 6.0,
        tau_pre: float = 20.0,    # janela de potenciação (ms)
        tau_post: float = 20.0,   # janela de depressão (ms)
        a_plus: float = 0.02,     # passo de potenciação
        a_minus: float | None = None,  # passo de depressão (negativo)
        tau_syn: float = 10.0,    # decaimento da corrente sináptica (ms)
        plastic: bool = True,
        dt: float = 0.1,
    ) -> None:
        self.n_pre = n_pre
        self.n_post = n_post
        self.w = np.full((n_pre, n_post), w_init, dtype=np.float64) \
            if np.isscalar(w_init) else np.array(w_init, dtype=np.float64).reshape(n_pre, n_post)
        self.w_max = w_max
        self.a_plus = a_plus
        # Convenção clássica: depressão ~5% mais forte que potenciação, garante
        # estabilidade (pesos não explodem com atividade descorrelacionada).
        self.a_minus = a_minus if a_minus is not None else -a_plus * 1.05
        self.plastic = plastic
        self.dt = dt

        self._decay_pre = np.exp(-dt / tau_pre)
        self._decay_post = np.exp(-dt / tau_post)
        self._decay_syn = np.exp(-dt / tau_syn)

        self.reset_state()

    def reset_state(self) -> None:
        """Zera traços e corrente sináptica (NÃO mexe nos pesos aprendidos)."""
        self.a_pre = np.zeros(self.n_pre, dtype=np.float64)
        self.a_post = np.zeros(self.n_post, dtype=np.float64)
        self.i_syn = np.zeros(self.n_post, dtype=np.float64)

    def propagate(self, pre_spikes: np.ndarray) -> np.ndarray:
        """Avança a corrente sináptica um passo e retorna I (nA) para o pós.

        `pre_spikes`: máscara booleana (n_pre,) de quem disparou neste passo.
        """
        self.i_syn *= self._decay_syn
        if np.any(pre_spikes):
            # Soma os pesos das linhas (pré) que dispararam, para cada pós.
            self.i_syn += self.w[pre_spikes, :].sum(axis=0)
        return self.i_syn

    def update(self, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> None:
        """Aplica a regra STDP dado quem disparou neste passo (pré e pós)."""
        if not self.plastic:
            return

        # Decaimento contínuo dos traços.
        self.a_pre *= self._decay_pre
        self.a_post *= self._decay_post

        # Disparo do pré: depressão (usa a_post atual) + reforça traço pré.
        if np.any(pre_spikes):
            self.w[pre_spikes, :] = np.clip(
                self.w[pre_spikes, :] + self.a_post[None, :], 0.0, self.w_max
            )
            self.a_pre[pre_spikes] += self.a_plus

        # Disparo do pós: potenciação (usa a_pre atual) + reforça traço pós.
        if np.any(post_spikes):
            self.w[:, post_spikes] = np.clip(
                self.w[:, post_spikes] + self.a_pre[:, None], 0.0, self.w_max
            )
            self.a_post[post_spikes] += self.a_minus
