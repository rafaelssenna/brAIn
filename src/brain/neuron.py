"""Modelos de neurônio — M1.

Implementação do zero (NumPy) do neurônio Leaky Integrate-and-Fire (LIF),
o ponto de partida do brAIn. Ver docs/04-fundacoes-matematicas.md, seção 1.1.

A EDO da membrana:

    tau_m * dV/dt = -(V - V_rest) + R * I(t)

Integração de Euler explícito, passo dt:

    V_{k+1} = V_k + (dt/tau_m) * [ -(V_k - V_rest) + R * I_k ]

Regra de disparo: se V >= V_threshold, emite spike, V <- V_reset, e o neurônio
fica em período refratário (não integra) por `refractory` ms.

Unidades (consistentes): tempo em ms, V em mV, R em MΩ, I em nA
(MΩ * nA = mV). Assim dt/tau_m é adimensional e R*I sai em mV.
"""

from __future__ import annotations

import numpy as np


class LIFPopulation:
    """Uma população de N neurônios LIF independentes (N=1 => um neurônio).

    Vetorizado: o estado é um array (N,), então simular 1 ou 10.000 neurônios
    é o mesmo código. Isso já prepara o terreno para o M2 (redes/STDP).
    """

    def __init__(
        self,
        n: int = 1,
        tau_m: float = 10.0,       # constante de tempo da membrana (ms)
        v_rest: float = -65.0,     # potencial de repouso (mV)
        v_reset: float = -65.0,    # potencial pós-disparo (mV)
        v_threshold: float = -50.0,  # limiar de disparo (mV)
        r_m: float = 10.0,         # resistência da membrana (MΩ)
        refractory: float = 2.0,   # período refratário (ms)
        dt: float = 0.1,           # passo de integração (ms)
    ) -> None:
        if n < 1:
            raise ValueError("n deve ser >= 1")
        if tau_m <= 0 or dt <= 0:
            raise ValueError("tau_m e dt devem ser > 0")
        if dt > tau_m:
            # Euler explícito fica instável se o passo for grande demais.
            raise ValueError("dt deve ser <= tau_m para estabilidade do Euler")

        self.n = n
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.r_m = r_m
        self.refractory = refractory
        self.dt = dt

        self.reset_state()

    def reset_state(self) -> None:
        """Volta a população ao estado inicial (repouso, t=0)."""
        self.v = np.full(self.n, self.v_rest, dtype=np.float64)
        # Tempo (ms) até o qual cada neurônio ainda está refratário.
        self._refrac_until = np.zeros(self.n, dtype=np.float64)
        self.t = 0.0

    def step(self, current: np.ndarray | float) -> np.ndarray:
        """Avança um passo dt. Retorna máscara booleana (N,) de quem disparou.

        `current` é a corrente de entrada I em nA: escalar ou array (N,).
        """
        i = np.broadcast_to(np.asarray(current, dtype=np.float64), (self.n,))

        self.t += self.dt
        # Neurônios fora do período refratário integram a EDO.
        active = self.t >= self._refrac_until

        dv = (self.dt / self.tau_m) * (-(self.v - self.v_rest) + self.r_m * i)
        self.v = np.where(active, self.v + dv, self.v_reset)

        # Disparos: cruzou o limiar (e está ativo).
        spikes = active & (self.v >= self.v_threshold)
        # Reset e início do período refratário para quem disparou.
        self.v = np.where(spikes, self.v_reset, self.v)
        self._refrac_until = np.where(spikes, self.t + self.refractory, self._refrac_until)

        return spikes

    def run(self, currents: np.ndarray, record_voltage: bool = True):
        """Simula uma sequência de correntes.

        `currents` tem shape (T,) [mesma corrente p/ todos] ou (T, N).
        Retorna (spike_times_por_neuronio, voltagens).

        - spikes: lista de N arrays, cada um com os tempos (ms) de disparo.
        - voltages: array (T, N) com o potencial de membrana (ou None).
        """
        currents = np.asarray(currents, dtype=np.float64)
        n_steps = currents.shape[0]

        voltages = np.empty((n_steps, self.n)) if record_voltage else None
        spike_steps = [[] for _ in range(self.n)]

        for k in range(n_steps):
            # step() já faz broadcast tanto de currents[k] escalar (T,) quanto
            # de uma linha (T, N) -> não precisa distinguir os dois casos.
            i_k = currents[k]
            spikes = self.step(i_k)
            if record_voltage:
                # Registra o potencial; marca o disparo com um "pico" visual no limiar
                # para o traço não parecer cortado bruscamente.
                v_now = self.v.copy()
                v_now[spikes] = self.v_threshold + 20.0
                voltages[k] = v_now
            for idx in np.nonzero(spikes)[0]:
                spike_steps[idx].append(self.t)

        spike_times = [np.asarray(s) for s in spike_steps]
        return spike_times, voltages


def fi_curve_theory(
    currents_na: np.ndarray,
    tau_m: float = 10.0,
    v_rest: float = -65.0,
    v_reset: float = -65.0,
    v_threshold: float = -50.0,
    r_m: float = 10.0,
    refractory: float = 2.0,
) -> np.ndarray:
    """Frequência de disparo teórica (Hz) de um LIF sob corrente constante.

    Para corrente supralimiar, o intervalo entre disparos é

        T = tau_m * ln[ (V_inf - V_reset) / (V_inf - V_threshold) ]

    com V_inf = V_rest + R*I (potencial de equilíbrio). A taxa é
    f = 1000 / (T + refractory)  [Hz, com tempos em ms].
    Abaixo do limiar (V_inf <= V_threshold) o neurônio nunca dispara: f = 0.
    """
    i = np.asarray(currents_na, dtype=np.float64)
    v_inf = v_rest + r_m * i
    rate = np.zeros_like(i)
    fires = v_inf > v_threshold
    t_isi = tau_m * np.log(
        (v_inf[fires] - v_reset) / (v_inf[fires] - v_threshold)
    )
    rate[fires] = 1000.0 / (t_isi + refractory)
    return rate
