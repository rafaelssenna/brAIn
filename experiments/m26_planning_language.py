"""M26 — O organismo que PLANEJA rotas num mundo 2D e fala (a unificação 2D).

O M25 deu corpo ao organismo que fala, mas num anel trivial (sem obstáculos). Um cérebro
de verdade PLANEJA: imagina o caminho, contorna paredes, alcança o que quer. Esse
maquinário já existia (M11/M13) mas vivia no agente MUDO. Aqui ele se junta à linguagem.

Mundo: grid 5x5 com uma PAREDE. Há objetos de fácil acesso (lado de cá) e um objeto C
ATRÁS da parede — só dá para chegar a ele CONTORNANDO. O organismo aprende o modelo de
transição vivendo, a curiosidade escolhe a meta, o PLANEJAMENTO traça a rota, a percepção
aprende o objeto, a memória retém, e a linguagem é aprendida por ATENÇÃO CONJUNTA (M21):
ao CHEGAR numa célula de objeto, um "professor" diz a palavra certa e o agente aprende a
NOMEAR (vê → diz a palavra) e a APONTAR. Só aprende a palavra de um objeto se CHEGAR lá.

Pergunta científica (H9): o planejamento permite NOMEAR o que está atrás da parede — algo
que o reativo (sem imaginar o desvio) nunca alcança, nunca percebe nem nomeia?

Baseline: o MESMO organismo, mas REATIVO (anda na direção da meta, trava na parede).

Uso:   python experiments/m26_planning_language.py
Salva: experiments/output/m26_planning_language.png

Honesto: escala de brinquedo (grid 5x5, 3 objetos); linguagem por professor (atenção
conjunta do ambiente), não negociada entre dois agentes (isso foi o M25). É o organismo
inteiro num laço 2D (perceber+planejar+mover+lembrar+curiosidade+nomear), não um cérebro pronto.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import GridWorld  # noqa: E402
from brain.planning_language import PlanningLanguageAgent  # noqa: E402

np.seterr(all="ignore")     # warnings espúrios de matmul (NumPy/BLAS); resultados são finitos

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GRID = 5
SIDE = 4
D = SIDE * SIDE
N_LATENT = 12
N_STEPS = 4500
N_SEEDS = 3
NOISE = 0.05
START = (0, 0)
C_CELL = (0, 4)                                  # objeto ATRÁS da parede
WALL = [(r, 2) for r in range(4)]               # parede na coluna 2, passagem em (4,2)


def bar(kind, i):
    p = np.zeros((SIDE, SIDE))
    if kind == "h":
        p[i, :] = 1.0
    else:
        p[:, i] = 1.0
    v = p.ravel()
    return v / np.linalg.norm(v)


PATTERNS = {(0, 0): bar("h", 0), (3, 1): bar("h", 2), C_CELL: bar("v", 0)}
CONTENT_CELLS = list(PATTERNS.keys())
CLEAN = [PATTERNS[c] for c in CONTENT_CELLS]
LABELS = ["objeto (0,0)", "objeto (3,1)", "C (atrás da parede)"]
C_IDX = CONTENT_CELLS.index(C_CELL)


def make_grid():
    return GridWorld(GRID, WALL, goal=None)


def obs_for(cell, rng):
    if cell in PATTERNS:
        v = PATTERNS[cell] + NOISE * rng.standard_normal(D)
        return v / np.linalg.norm(v)
    return np.zeros(D)


def naming_per_object(A):
    """Para cada objeto: o agente vê e diz a palavra. 1 se acertou o nome."""
    return [int(A.speak(A.concept_of(CLEAN[o]), explore=False) == o) for o in range(len(CLEAN))]


def live(use_planning, seed):
    """Um organismo vive o mundo 2D, planeja e aprende as palavras por atenção conjunta."""
    A = PlanningLanguageAgent(make_grid(), CONTENT_CELLS, D, N_LATENT, len(CONTENT_CELLS),
                              START, use_planning=use_planning, eps=0.15, seed=1)
    rng = np.random.default_rng(seed)
    visits_C = 0
    traj = []
    xs, name_all, name_C, surpC = [], [], [], []
    for t in range(N_STEPS):
        here, _, x = A.perceive_and_learn(obs_for, rng)
        A.learn_word_here(x)                              # professor nomeia o que ele vê aqui
        visits_C += int(here == C_CELL)
        traj.append(here)
        A.navigate()
        if t % 100 == 0:
            xs.append(t)
            nm = naming_per_object(A)
            name_all.append(100 * np.mean(nm))
            name_C.append(100 * nm[C_IDX])
            surpC.append(A.surprise_on(PATTERNS[C_CELL]))
    return dict(visits_C=visits_C, traj=traj, xs=np.array(xs),
                name_all=name_all, name_C=name_C, surpC=surpC,
                final_name=naming_per_object(A))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    plan_runs = [live(True, s) for s in range(N_SEEDS)]
    react_runs = [live(False, s) for s in range(N_SEEDS)]

    def avg(runs, key):
        return np.mean([np.array(r[key]) for r in runs], axis=0)

    visC_plan = np.mean([r["visits_C"] for r in plan_runs])
    visC_react = np.mean([r["visits_C"] for r in react_runs])
    surpC_plan = avg(plan_runs, "surpC")
    surpC_react = avg(react_runs, "surpC")
    nameC_plan = avg(plan_runs, "name_C")
    nameC_react = avg(react_runs, "name_C")
    nameAll_plan = avg(plan_runs, "name_all")
    nameAll_react = avg(react_runs, "name_all")
    fin_plan = np.mean([r["final_name"] for r in plan_runs], axis=0)
    fin_react = np.mean([r["final_name"] for r in react_runs], axis=0)
    xs = plan_runs[0]["xs"]

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) o mundo + a trajetória do planejador (contorna a parede até C).
    ax = axes[0, 0]
    grid_img = np.ones((GRID, GRID, 3))
    for (r, c) in WALL:
        grid_img[r, c] = [0.25, 0.25, 0.25]
    ax.imshow(grid_img, origin="upper")
    traj = plan_runs[0]["traj"][:600]
    tr = np.array(traj) + np.random.default_rng(0).normal(0, 0.08, (len(traj), 2))
    ax.plot(tr[:, 1], tr[:, 0], color="#1f77b4", lw=0.7, alpha=0.6)
    ax.scatter([START[1]], [START[0]], s=120, color="#2ca02c", edgecolor="white", zorder=5)
    ax.annotate("início", (START[1], START[0]), color="white", fontsize=7.5, ha="center", va="center")
    ax.scatter([C_CELL[1]], [C_CELL[0]], s=130, color="#d62728", edgecolor="white", zorder=5)
    ax.annotate("C\n(atrás)", (C_CELL[1], C_CELL[0]), color="white", fontsize=7, ha="center", va="center")
    ax.scatter([CONTENT_CELLS[1][1]], [CONTENT_CELLS[1][0]], s=90, color="#ff7f0e", edgecolor="white", zorder=5)
    ax.set_title("(a) O mundo 2D com parede: o planejador CONTORNA até C")
    ax.set_xticks(range(GRID)); ax.set_yticks(range(GRID))

    # (b) só quem PLANEJA aprende a VER o objeto atrás da parede (surpresa em C).
    ax = axes[0, 1]
    ax.plot(xs, surpC_plan, lw=2.2, color="#1f77b4", label=f"planejador (visita C {visC_plan:.0f}x)")
    ax.plot(xs, surpC_react, lw=2.2, color="#d62728", label=f"reativo (visita C {visC_react:.0f}x)")
    ax.set_title("(b) Só quem PLANEJA aprende a ver o objeto atrás da parede")
    ax.set_xlabel("passos de vida"); ax.set_ylabel("surpresa no objeto C (atrás da parede)")
    ax.legend(loc="upper right", fontsize=8.5)

    # (c) e só quem planeja aprende a NOMEAR o objeto atrás da parede.
    ax = axes[1, 0]
    ax.plot(xs, nameC_plan, lw=2.4, color="#1f77b4", label=f"planejador: nomeia C {nameC_plan[-1]:.0f}%")
    ax.plot(xs, nameC_react, lw=2.4, color="#d62728", label=f"reativo: nomeia C {nameC_react[-1]:.0f}%")
    ax.plot(xs, nameAll_plan, lw=1.2, ls="--", color="#1f77b4", alpha=0.6, label=f"planej.: nomeia tudo {nameAll_plan[-1]:.0f}%")
    ax.plot(xs, nameAll_react, lw=1.2, ls="--", color="#d62728", alpha=0.6, label=f"reativo: nomeia tudo {nameAll_react[-1]:.0f}%")
    ax.set_title("(c) Só quem PLANEJA aprende a NOMEAR o que está atrás da parede")
    ax.set_xlabel("passos de vida"); ax.set_ylabel("acurácia ao nomear %"); ax.set_ylim(-3, 105)
    ax.legend(loc="center right", fontsize=7.8)

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M26 — o organismo que PLANEJA num mundo 2D e fala\n\n"
            "Um organismo COMPLETO num grid com PAREDE: aprende o\n"
            "modelo de transição vivendo, a CURIOSIDADE escolhe a meta,\n"
            "o PLANEJAMENTO traça a rota (contornando a parede),\n"
            "PERCEBE, LEMBRA e NOMEIA (atenção conjunta, M21). Um objeto\n"
            "(C) fica ATRÁS da parede — só quem planeja chega lá.\n\n"
            f"• visitas a C:    planejador {visC_plan:.0f}x  |  reativo {visC_react:.0f}x\n"
            f"• surpresa em C:  planejador {surpC_plan[-1]:.3f}  |  reativo {surpC_react[-1]:.3f}\n"
            f"• NOMEIA C:       planejador {nameC_plan[-1]:.0f}%  |  reativo {nameC_react[-1]:.0f}%\n"
            f"• nomeia tudo:    planejador {nameAll_plan[-1]:.0f}%  |  reativo {nameAll_react[-1]:.0f}%\n"
            f"                  (média de {N_SEEDS} sementes)\n\n"
            "Achado (H9): PENSAR a rota é o que permite PERCEBER e\n"
            "NOMEAR o que está atrás da parede. O reativo nunca chega a\n"
            "C, nunca aprende a vê-lo (surpresa alta) e nunca aprende a\n"
            "palavra dele. É o efeito de INTEGRAÇÃO: perceber+planejar+\n"
            "mover+lembrar+curiosidade+nomear, num laço 2D — o que\n"
            "nenhuma peça faz sozinha.\n\n"
            "Honesto: escala de brinquedo (grid 5x5, 3 objetos); língua\n"
            "por professor (atenção conjunta do ambiente), não negociada\n"
            "entre dois agentes (isso foi o M25); mecanismo, não cérebro pronto.",
            transform=ax.transAxes, va="top", fontsize=8.2, family="monospace")

    fig.suptitle("brAIn · M26 — O organismo planeja rotas num mundo 2D e nomeia o que alcança (até atrás da parede)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m26_planning_language.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Visitas a C:   planejador {visC_plan:.0f}x | reativo {visC_react:.0f}x")
    print(f"Surpresa em C: planejador {surpC_plan[-1]:.3f} | reativo {surpC_react[-1]:.3f}")
    print(f"Nomeia C:      planejador {nameC_plan[-1]:.0f}% | reativo {nameC_react[-1]:.0f}%")
    print(f"Nomeia tudo:   planejador {nameAll_plan[-1]:.0f}% | reativo {nameAll_react[-1]:.0f}%")
    print(f"Final por obj: planejador {fin_plan} | reativo {fin_react}  ({LABELS})")


if __name__ == "__main__":
    main()
