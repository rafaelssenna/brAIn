"""M7 — O laço integrado (Fase 2, active inference mínima).

Um ÚNICO organismo: corpo num anel de lugares (M3) + modelo de mundo predictive
coding (M4) + curiosidade que guia a NAVEGAÇÃO física (M5). O agente precisa SE
MOVER até onde vale a pena aprender — a atenção virou física.

Demonstra (hipótese H3): o agente integrado aprende o modelo de mundo e navega de
forma competente — domina os lugares aprendíveis e EVITA fisicamente o ruído —
melhor que um corpo que passeia ao acaso.

Uso:   python experiments/m7_integrated.py
Salva: experiments/output/m7_integrated.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import RingWorld, IntegratedAgent  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
N_LOC = 8
NOISE_LOCS = (3, 6)
N_STEPS = 5000
EVAL_EVERY = 100


def run(policy, seed=SEED):
    world = RingWorld(n_locations=N_LOC, noise_locs=NOISE_LOCS, seed=1)
    agent = IntegratedAgent(world, policy=policy, seed=seed)
    rng = np.random.default_rng(seed)

    learn = world.learnable_locs()
    err0 = {l: agent.pc.prediction_error(world.patterns[l]) for l in learn}
    occupancy = np.zeros(N_LOC)
    positions = np.zeros(N_STEPS, dtype=int)
    checkpoints, mastery = [], []

    for t in range(N_STEPS):
        here, _ = agent.step(rng)
        occupancy[here] += 1
        positions[t] = here
        if t % EVAL_EVERY == 0:
            m = agent.mastery()
            tot = np.mean([max(0.0, 1.0 - m[l] / (err0[l] + 1e-9)) for l in learn])
            checkpoints.append(t); mastery.append(100.0 * tot)

    noise_time = 100.0 * sum(occupancy[l] for l in NOISE_LOCS) / N_STEPS
    return dict(world=world, occupancy=occupancy, positions=positions,
                checkpoints=np.array(checkpoints), mastery=np.array(mastery),
                noise_time=noise_time)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cur = run("curious")
    ran = run("random")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

    # (a) Domínio do modelo de mundo ao longo da vida.
    ax = axes[0, 0]
    ax.plot(cur["checkpoints"], cur["mastery"], lw=2.2, color="#2ca02c",
            label=f"integrado/curioso (ruído: {cur['noise_time']:.0f}% do tempo)")
    ax.plot(ran["checkpoints"], ran["mastery"], lw=2.0, color="#888888", ls="--",
            label=f"passeio aleatório (ruído: {ran['noise_time']:.0f}%)")
    ax.set_title("(a) Aprende o modelo de mundo — corpo, previsão e curiosidade juntos")
    ax.set_xlabel("experiência (passo)"); ax.set_ylabel("domínio dos lugares (%)")
    ax.legend(loc="lower right", fontsize=8.5); ax.set_ylim(-3, 103)

    # (b) Ocupação por lugar (curioso): dwell no aprendível, foge do ruído.
    ax = axes[0, 1]
    colors = ["#d62728" if l in NOISE_LOCS else "#2ca02c" for l in range(N_LOC)]
    ax.bar(range(N_LOC), 100 * cur["occupancy"] / N_STEPS, color=colors)
    ax.set_title("(b) Onde o corpo fica (vermelho = ruído): evita o inaprendível")
    ax.set_xlabel("lugar (anel)"); ax.set_ylabel("tempo ali (%)")
    ax.set_xticks(range(N_LOC))

    # (c) Trajetória no anel: navegação deliberada vs difusão.
    ax = axes[1, 0]
    seg = slice(0, 1200)
    ax.plot(np.arange(N_STEPS)[seg], cur["positions"][seg], lw=1.0, color="#2ca02c",
            label="curioso (navega e fica)")
    ax.plot(np.arange(N_STEPS)[seg], ran["positions"][seg], lw=0.8, color="#888888",
            alpha=0.7, label="aleatório (difunde)")
    for l in NOISE_LOCS:
        ax.axhline(l, color="#d62728", lw=0.6, ls=":", alpha=0.6)
    ax.set_title("(c) Trajetória: curioso navega de propósito (linhas pontilhadas = ruído)")
    ax.set_xlabel("passo"); ax.set_ylabel("lugar"); ax.set_yticks(range(N_LOC))
    ax.legend(loc="upper right", fontsize=8)

    # (d) Resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M7 — o laço integrado (active inference mínima)\n\n"
            "Um organismo só: CORPO (M3) + MODELO DE MUNDO (M4)\n"
            "+ CURIOSIDADE (M5) guiando a navegação física.\n\n"
            f"• domínio final  integrado={cur['mastery'][-1]:.0f}%  "
            f"aleatório={ran['mastery'][-1]:.0f}%\n"
            f"• tempo no ruído integrado={cur['noise_time']:.0f}%  "
            f"aleatório={ran['noise_time']:.0f}%\n\n"
            "O corpo torna a atenção CONSEQUENTE: navegar é caro,\n"
            "então alocar bem importa (ao contrário do M5 'teletransporte').\n\n"
            "Honesto: ainda rate-based (sem spiking, vem no M10);\n"
            "e o esquecimento da Fase 1 segue aberto (alvo do M8).",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M7 — O organismo integrado: corpo + previsão + curiosidade num laço só",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m7_integrated.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"Domínio final  integrado={cur['mastery'][-1]:.0f}%  aleatório={ran['mastery'][-1]:.0f}%")
    print(f"Tempo no ruído  integrado={cur['noise_time']:.0f}%  aleatório={ran['noise_time']:.0f}%")


if __name__ == "__main__":
    main()
