"""M13 — O organismo integrado (a costura final da Fase 2).

Tudo num laço só: corpo num grid com PAREDE + planejamento (M11) + predictive
coder hierárquico (M9) + curiosidade (M5) + memória/replay (M8).

Mundo: grid 5x5 com parede; lugares com conteúdo aprendível A, B (lado esquerdo)
e C (do outro lado da parede — só dá para chegar CONTORNANDO). Mais um lugar de
ruído. A curiosidade escolhe a meta; o planejamento traça a rota; a hierarquia
aprende; a memória retém.

Resultado robusto (média de seeds): só quem PLANEJA alcança C — o reativo trava
na parede. É o efeito de integração que nenhuma peça reativa consegue sozinha.
Honesto: o domínio por-lugar é ruidoso (hierarquia profunda + dinâmica complexa);
por isso medimos visitas a C (limpo) e domínio médio sobre seeds.

Uso:   python experiments/m13_integrated.py
Salva: experiments/output/m13_integrated.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import GridWorld, IntegratedBrain  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GRID = 5
SIDE = 4
D = SIDE * SIDE
SIZES = [D, 12, 6]
N_STEPS = 4000
N_SEEDS = 3
START = (0, 0)
NOISE_CELL = (4, 4)
C_CELL = (0, 4)


def bar(kind, i):
    p = np.zeros((SIDE, SIDE))
    p[i, :] = 1.0 if kind == "h" else 0.0
    if kind == "v":
        p[:, i] = 1.0
    v = p.ravel()
    return v / np.linalg.norm(v)


PATTERNS = {(0, 0): bar("h", 0), (3, 1): bar("h", 2), C_CELL: bar("v", 0)}


def make_grid():
    return GridWorld(GRID, [(r, 2) for r in range(4)], goal=None)   # passagem em (4,2)


def run(use_planning, seed):
    grid = make_grid()
    brain = IntegratedBrain(grid, PATTERNS, [NOISE_CELL], START, SIZES,
                            use_planning=use_planning, use_memory=True, seed=0)
    rng = np.random.default_rng(seed)
    err0 = {c: brain.err_on(PATTERNS[c]) for c in PATTERNS}
    visits_C = 0
    positions = []
    checkpoints, total = [], []
    for t in range(N_STEPS):
        here, _ = brain.step(rng)
        visits_C += int(here == C_CELL)
        positions.append(here)
        if t % 80 == 0:
            m = brain.mastery()
            checkpoints.append(t)
            total.append(100 * np.mean([max(0.0, 1 - m[c] / (err0[c] + 1e-9)) for c in PATTERNS]))
    return dict(checkpoints=np.array(checkpoints), total=np.array(total),
                visits_C=100 * visits_C / N_STEPS, positions=positions)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    full = [run(True, s) for s in range(N_SEEDS)]
    nopl = [run(False, s) for s in range(N_SEEDS)]

    def stack(runs):
        return np.array([r["total"] for r in runs])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) mundo + trajetória de exemplo (integrado contorna a parede até C).
    ax = axes[0, 0]
    g = make_grid()
    img = np.zeros((GRID, GRID))
    for (r, c) in g.walls:
        img[r, c] = 1
    ax.imshow(img, cmap="Greys", vmin=0, vmax=1)
    for c, name in zip(PATTERNS, ["A", "B", "C"]):
        ax.scatter(c[1], c[0], s=260, color="#2ca02c", zorder=4)
        ax.text(c[1], c[0], name, ha="center", va="center", color="w", fontweight="bold")
    ax.scatter(NOISE_CELL[1], NOISE_CELL[0], s=240, marker="X", color="#d62728", zorder=4)
    pos = np.array(full[0]["positions"][:800])
    j = np.random.default_rng(0).uniform(-.12, .12, pos.shape)
    ax.plot(pos[:, 1] + j[:, 1], pos[:, 0] + j[:, 0], lw=0.7, color="#1f77b4", alpha=0.7)
    ax.scatter(START[1], START[0], s=120, marker="s", color="k", zorder=5)
    ax.set_title("(a) A,B (esq) · C atrás da parede · X=ruído · ■=início")
    ax.set_xticks([]); ax.set_yticks([])

    # (b) visitas a C (robusto): só o planejador chega.
    ax = axes[0, 1]
    vf = [r["visits_C"] for r in full]; vn = [r["visits_C"] for r in nopl]
    bars = ax.bar(["integrado\n(planeja)", "sem\nplanejamento"],
                  [np.mean(vf), np.mean(vn)],
                  yerr=[np.std(vf), np.std(vn)], capsize=5, color=["#2ca02c", "#ff7f0e"])
    for b, v in zip(bars, [np.mean(vf), np.mean(vn)]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"(b) Visitas a C, atrás da parede (média de {N_SEEDS} seeds)")
    ax.set_ylabel("% do tempo em C")

    # (c) domínio total (média ± desvio sobre seeds).
    ax = axes[1, 0]
    t = full[0]["checkpoints"]
    for runs, color, lbl in [(full, "#2ca02c", "integrado"), (nopl, "#ff7f0e", "sem planejamento")]:
        S = stack(runs); m = S.mean(0); sd = S.std(0)
        ax.plot(t, m, lw=2.2, color=color, label=lbl)
        ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.18)
    ax.set_title(f"(c) Domínio total do mundo (média de {N_SEEDS} seeds)")
    ax.set_xlabel("experiência (passo)"); ax.set_ylabel("domínio (%)")
    ax.legend(loc="lower right", fontsize=9); ax.set_ylim(-3, 103)

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M13 — o organismo integrado (a costura final)\n\n"
            "Num laço só: CORPO + PLANEJAMENTO + HIERARQUIA\n"
            "+ CURIOSIDADE + MEMÓRIA.\n\n"
            "Divisão de trabalho que EMERGE:\n"
            "  • curiosidade decide PARA ONDE ir\n"
            "  • planejamento traça COMO chegar (contorna a parede)\n"
            "  • hierarquia APRENDE o que há lá\n"
            "  • memória RETÉM o aprendido\n\n"
            f"Visitas a C (atrás da parede):\n"
            f"  integrado {np.mean(vf):.1f}%  vs  sem planejam. {np.mean(vn):.1f}%\n"
            f"Domínio total final:\n"
            f"  integrado {stack(full)[:,-1].mean():.0f}%  vs  sem planejam. {stack(nopl)[:,-1].mean():.0f}%\n\n"
            "Honesto: o domínio por-lugar é ruidoso (hierarquia\n"
            "profunda + dinâmica complexa). O efeito ROBUSTO é\n"
            "que só planejando se alcança C. A H1 costurada —\n"
            "em miniatura, rate-based; Fase 3 = cognição.",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M13 — O organismo integrado: tudo num laço só (a H1 costurada)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m13_integrated.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"Visitas a C:  integrado {np.mean(vf):.1f}% (±{np.std(vf):.1f})  "
          f"sem planejam. {np.mean(vn):.1f}% (±{np.std(vn):.1f})")
    print(f"Domínio total final:  integrado {stack(full)[:,-1].mean():.0f}%  "
          f"sem planejam. {stack(nopl)[:,-1].mean():.0f}%")


if __name__ == "__main__":
    main()
