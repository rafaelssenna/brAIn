"""M1 — O primeiro neurônio vivo.

Roda o neurônio LIF e gera os gráficos canônicos que mostram a matemática
"respirando":

  (a) Potencial de membrana ao longo do tempo, sob corrente em degrau:
      o potencial sobe, dispara, reseta, repete.
  (b) Curva f-I: frequência de disparo vs corrente — simulado vs teoria.
      (validação: tem de bater com a fórmula analítica)
  (c) Raster de uma pequena população com correntes heterogêneas:
      cada neurônio com seu ritmo próprio.

Uso:
    python experiments/m1_living_neuron.py

Salva a figura em experiments/output/m1_living_neuron.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")  # backend sem display (roda no terminal)
import matplotlib.pyplot as plt

# Permite rodar direto sem instalar o pacote (adiciona src/ ao path).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LIFPopulation, fi_curve_theory  # noqa: E402

SEED = 42  # reprodutibilidade (regra do roadmap)
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def panel_membrane(ax) -> None:
    """(a) Um neurônio, corrente em degrau, traço do potencial."""
    dt = 0.1
    neuron = LIFPopulation(n=1, dt=dt)

    # 200 ms: 50 ms sem corrente, 150 ms com 2.0 nA (supralimiar).
    t_total = 200.0
    n_steps = int(t_total / dt)
    currents = np.zeros(n_steps)
    currents[int(50 / dt):] = 2.0

    spikes, voltages = neuron.run(currents)
    time = np.arange(n_steps) * dt

    ax.plot(time, voltages[:, 0], lw=1.0, color="#1f77b4")
    ax.axhline(neuron.v_threshold, ls="--", lw=0.8, color="#d62728", label="limiar")
    ax.axhline(neuron.v_rest, ls=":", lw=0.8, color="gray", label="repouso")
    ax.axvspan(50, t_total, color="#ffe9b3", alpha=0.4, zorder=0, label="corrente ON (2 nA)")
    ax.set_title("(a) Membrana de um neurônio: integra, dispara, reseta")
    ax.set_xlabel("tempo (ms)")
    ax.set_ylabel("V (mV)")
    ax.legend(loc="upper right", fontsize=7)
    n_sp = len(spikes[0])
    ax.text(0.02, 0.95, f"{n_sp} disparos", transform=ax.transAxes,
            fontsize=8, va="top")


def panel_fi_curve(ax) -> None:
    """(b) Curva f-I: simulado vs teoria analítica."""
    dt = 0.1
    sim_currents = np.linspace(0.0, 5.0, 26)
    sim_rates = []
    t_total = 1000.0  # 1 s por ponto, para estimar a taxa em Hz
    n_steps = int(t_total / dt)

    for i_amp in sim_currents:
        neuron = LIFPopulation(n=1, dt=dt)
        spikes, _ = neuron.run(np.full(n_steps, i_amp), record_voltage=False)
        sim_rates.append(len(spikes[0]) / (t_total / 1000.0))  # Hz

    theory_currents = np.linspace(0.0, 5.0, 400)
    theory_rates = fi_curve_theory(theory_currents)

    ax.plot(theory_currents, theory_rates, lw=1.5, color="#2ca02c", label="teoria")
    ax.scatter(sim_currents, sim_rates, s=18, color="#1f77b4", zorder=3, label="simulado")
    ax.set_title("(b) Curva f-I: simulação valida a teoria")
    ax.set_xlabel("corrente I (nA)")
    ax.set_ylabel("taxa de disparo (Hz)")
    ax.legend(loc="upper left", fontsize=7)


def panel_raster(ax) -> None:
    """(c) Raster de uma população heterogênea."""
    rng = np.random.default_rng(SEED)
    dt = 0.1
    n = 20
    pop = LIFPopulation(n=n, dt=dt)

    t_total = 300.0
    n_steps = int(t_total / dt)
    # Cada neurônio recebe uma corrente base diferente (1.6 a 3.0 nA) + ruído.
    base = np.linspace(1.6, 3.0, n)
    currents = base[None, :] + rng.normal(0, 0.3, size=(n_steps, n))

    spikes, _ = pop.run(currents, record_voltage=False)

    for idx, sp in enumerate(spikes):
        ax.scatter(sp, np.full_like(sp, idx), s=4, color="#1f77b4")
    ax.set_title("(c) População: 20 neurônios, ritmos próprios")
    ax.set_xlabel("tempo (ms)")
    ax.set_ylabel("neurônio #")
    ax.set_ylim(-1, n)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    panel_membrane(axes[0])
    panel_fi_curve(axes[1])
    panel_raster(axes[2])
    fig.suptitle("brAIn · M1 — O primeiro neurônio vivo (Leaky Integrate-and-Fire)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = os.path.join(OUT_DIR, "m1_living_neuron.png")
    fig.savefig(out_path, dpi=130)
    print(f"Figura salva em: {out_path}")


if __name__ == "__main__":
    main()
