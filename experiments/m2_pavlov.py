"""M2 — Condicionamento pavloviano in silico.

Demonstra APRENDIZADO ASSOCIATIVO sem nenhum backpropagation, só com STDP.

Montagem (o "cachorro de Pavlov" em neurônios):

    CS  (sino, estímulo neutro) ---[ sinapse PLÁSTICA, começa fraca ]---> OUT
    US  (comida, incondicionado)--[ sinapse FIXA e forte ]------------->  (resposta)

  - No início, o sino (CS) sozinho NÃO faz o OUT disparar (sinapse fraca).
  - A comida (US) SEMPRE faz o OUT disparar (sinapse forte e fixa).
  - Treino: apresenta CS e logo depois US, repetidamente. Como o CS dispara
    pouco ANTES do OUT (que o US obriga a disparar), o STDP POTENCIA a sinapse
    CS->OUT a cada par.
  - Depois de muitos pares, o sino SOZINHO já dispara o OUT: resposta condicionada.

Controle científico: a condição "não-pareada" (CS sem US) NÃO deve aprender —
prova que o aprendizado é associativo, não mero efeito da atividade.

Uso:   python experiments/m2_pavlov.py
Salva: experiments/output/m2_pavlov.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LIFPopulation, STDPConnection  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

DT = 0.1                 # ms
TRIAL_MS = 200.0         # duração de cada tentativa
N_TRIALS = 60            # número de tentativas de treino
N_STEPS = int(TRIAL_MS / DT)

# Tempos de disparo dentro de uma tentativa (ms).
CS_TIMES = [50.0, 52.0, 54.0, 56.0, 58.0]   # rajada do sino
US_TIMES = [62.0, 64.0, 66.0]               # comida, logo após o sino


def _spike_steps(times):
    return {int(round(t / DT)) for t in times}


def build():
    out = LIFPopulation(n=1, dt=DT)
    # CS->OUT: plástica, começa fraca (sublimiar sozinha).
    cs = STDPConnection(1, 1, w_init=0.4, w_max=6.0, a_plus=0.02,
                        tau_syn=10.0, plastic=True, dt=DT)
    # US->OUT: fixa e forte (sempre dispara o OUT).
    us = STDPConnection(1, 1, w_init=5.0, plastic=False, tau_syn=10.0, dt=DT)
    return out, cs, us


def run_trial(out, cs, us, present_cs, present_us, learn=True, record=False):
    """Roda uma tentativa. Retorna (n_spikes_out, voltagem|None)."""
    out.reset_state()   # cada tentativa começa do zero (sem vazamento de estado)
    cs.reset_state()
    us.reset_state()
    cs_steps = _spike_steps(CS_TIMES) if present_cs else set()
    us_steps = _spike_steps(US_TIMES) if present_us else set()

    n_out = 0
    volt = np.empty(N_STEPS) if record else None
    for k in range(N_STEPS):
        pre_cs = np.array([k in cs_steps])
        pre_us = np.array([k in us_steps])
        i = cs.propagate(pre_cs) + us.propagate(pre_us)
        spikes = out.step(i)
        if learn:
            cs.update(pre_cs, spikes)   # só a sinapse do CS aprende
        if record:
            v = out.v[0]
            volt[k] = v if not spikes[0] else out.v_threshold + 20.0
        n_out += int(spikes[0])
    return n_out, volt


def run_condition(paired: bool):
    """Roda o treino completo. Retorna trajetória do peso CS->OUT e do nº de
    disparos do OUT por tentativa."""
    out, cs, us = build()
    weights, out_spikes = [], []
    for _ in range(N_TRIALS):
        n_out, _ = run_trial(out, cs, us, present_cs=True, present_us=paired)
        weights.append(cs.w[0, 0])
        out_spikes.append(n_out)
    return np.array(weights), np.array(out_spikes), (out, cs, us)


def test_cs_alone(out, cs, us):
    """Apresenta só o CS (sem aprender) e devolve (n_spikes, voltagem)."""
    return run_trial(out, cs, us, present_cs=True, present_us=False,
                     learn=False, record=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    # --- ANTES do treino: testa o sino sozinho num modelo virgem.
    out0, cs0, us0 = build()
    n_before, volt_before = test_cs_alone(out0, cs0, us0)

    # --- Treino pareado (CS+US) e controle não-pareado (CS sozinho).
    w_paired, spk_paired, trained = run_condition(paired=True)
    w_unpaired, spk_unpaired, _ = run_condition(paired=False)

    # --- DEPOIS do treino pareado: testa o sino sozinho.
    n_after, volt_after = test_cs_alone(*trained)

    time = np.arange(N_STEPS) * DT
    trials = np.arange(1, N_TRIALS + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # (a) Trajetória do peso CS->OUT.
    ax = axes[0]
    ax.plot(trials, w_paired, lw=1.8, color="#2ca02c", label="pareado (CS+US)")
    ax.plot(trials, w_unpaired, lw=1.8, color="#888888", ls="--",
            label="controle (CS sozinho)")
    ax.set_title("(a) O peso da sinapse CS→OUT aprende")
    ax.set_xlabel("tentativa de treino")
    ax.set_ylabel("peso w (nA)")
    ax.legend(loc="upper left", fontsize=8)

    # (b) Antes vs depois: o sino sozinho passa a disparar o OUT.
    ax = axes[1]
    ax.plot(time, volt_before, lw=1.0, color="#888888",
            label=f"ANTES: {n_before} disparos")
    ax.plot(time, volt_after, lw=1.0, color="#1f77b4",
            label=f"DEPOIS: {n_after} disparos")
    ax.axhline(out0.v_threshold, ls="--", lw=0.8, color="#d62728")
    for t in CS_TIMES:
        ax.axvline(t, color="#ffb000", lw=0.6, alpha=0.6)
    ax.set_title("(b) Resposta condicionada ao sino (CS sozinho)")
    ax.set_xlabel("tempo (ms)")
    ax.set_ylabel("V do OUT (mV)")
    ax.legend(loc="upper right", fontsize=8)

    # (c) Disparos do OUT por tentativa (a resposta condicionada emergindo).
    ax = axes[2]
    ax.plot(trials, spk_paired, lw=1.6, color="#2ca02c", marker="o", ms=3,
            label="pareado")
    ax.plot(trials, spk_unpaired, lw=1.6, color="#888888", ls="--",
            label="controle")
    ax.set_title("(c) Disparos do OUT por tentativa")
    ax.set_xlabel("tentativa de treino")
    ax.set_ylabel("nº de disparos do OUT")
    ax.legend(loc="center right", fontsize=8)

    fig.suptitle("brAIn · M2 — Condicionamento pavloviano via STDP (sem backprop)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = os.path.join(OUT_DIR, "m2_pavlov.png")
    fig.savefig(out_path, dpi=130)
    print(f"Figura salva em: {out_path}")
    print(f"Sino sozinho — ANTES do treino: {n_before} disparos do OUT")
    print(f"Sino sozinho — DEPOIS do treino: {n_after} disparos do OUT")
    print(f"Peso CS->OUT: {w_paired[0]:.2f} -> {w_paired[-1]:.2f} nA (pareado)")
    print(f"Peso CS->OUT: {w_unpaired[0]:.2f} -> {w_unpaired[-1]:.2f} nA (controle)")


if __name__ == "__main__":
    main()
