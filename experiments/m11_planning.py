"""M11 — Agência: planejar = "pensar" (Fase 2).

O agente APRENDE o modelo de mundo vivendo (o que cada ação faz), depois PENSA:
simula ações na imaginação para achar o caminho até o objetivo — incluindo desviar
de uma parede que ele nunca poderia contornar só reagindo. Hipótese H6: planejar
pelo modelo supera o agente reativo e mostra antecipação.

Demonstra:
  (a) labirinto com parede: o PLANEJADOR desvia e chega; o REATIVO trava na parede;
  (b) a IMAGINAÇÃO é fiel: o caminho imaginado (no modelo) == o caminho real;
  (c) taxa de sucesso e passos: planejador vs reativo em vários pares início→objetivo.

Uso:   python experiments/m11_planning.py
Salva: experiments/output/m11_planning.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import (GridWorld, WorldModel, explore_and_learn, plan,  # noqa: E402
                   run_planner, run_reactive)
from brain.planning import DELTAS  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIZE = 7


def make_world(goal):
    # Parede horizontal na linha 3, cobrindo colunas 0..4 (passagem nas colunas 5,6).
    walls = [(3, c) for c in range(5)]
    return GridWorld(SIZE, walls, goal)


def rollout(world, start, actions):
    s = start; traj = [s]
    for a in actions:
        s = world.step(s, a); traj.append(s)
    return traj


def imagine(model, start, actions):
    s = start; traj = [s]
    for a in actions:
        s = model.predict(s, a); traj.append(s)
    return traj


def draw_grid(ax, world):
    grid = np.zeros((SIZE, SIZE))
    for (r, c) in world.walls:
        grid[r, c] = 1.0
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(SIZE)); ax.set_yticks(range(SIZE))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_xticks(np.arange(-.5, SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-.5, SIZE, 1), minor=True)
    ax.grid(which="minor", color="#ccc", lw=0.5)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- Cenário de exemplo: cantos opostos atravessando a parede. ---
    start, goal = (6, 0), (0, 0)
    world = make_world(goal)
    model = WorldModel()
    explore_and_learn(world, model, n_steps=3000, rng=np.random.default_rng(1))

    actions = plan(model, start, goal)
    plan_traj = rollout(world, start, actions) if actions else [start]
    imag_traj = imagine(model, start, actions) if actions else [start]

    # reativo (registra a trajetória)
    s = start; react_traj = [s]
    for t in range(80):
        if s == goal:
            break
        dr, dc = goal[0] - s[0], goal[1] - s[1]
        a = (1 if dr > 0 else 0) if abs(dr) >= abs(dc) else (3 if dc > 0 else 2)
        sp = world.step(s, a)
        if sp == s:
            sp = world.step(s, int(rng.integers(4)))
        s = sp; react_traj.append(s)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) trajetórias.
    ax = axes[0, 0]; draw_grid(ax, world)
    pt = np.array(plan_traj); rt = np.array(react_traj)
    ax.plot(pt[:, 1], pt[:, 0], "-o", color="#2ca02c", ms=4, lw=1.6, label="planejador (imagina o desvio)")
    ax.plot(rt[:, 1], rt[:, 0], "-o", color="#d62728", ms=3, lw=1.0, alpha=0.7, label="reativo (trava na parede)")
    ax.scatter([start[1]], [start[0]], s=120, marker="s", color="k", zorder=5, label="início")
    ax.scatter([goal[1]], [goal[0]], s=200, marker="*", color="#ffb000", edgecolor="k", zorder=5, label="objetivo")
    ax.set_title("(a) Planejar vence a parede que reagir não vence")
    ax.legend(loc="upper right", fontsize=7.5)

    # (b) imaginação vs realidade (devem coincidir).
    ax = axes[0, 1]; draw_grid(ax, world)
    it = np.array(imag_traj)
    ax.plot(it[:, 1], it[:, 0], "-", color="#1f77b4", lw=4, alpha=0.4, label="imaginado (no modelo)")
    ax.plot(pt[:, 1], pt[:, 0], "--o", color="#2ca02c", ms=3, lw=1.2, label="real (executado)")
    ax.scatter([goal[1]], [goal[0]], s=200, marker="*", color="#ffb000", edgecolor="k", zorder=5)
    match = "idênticos" if imag_traj == plan_traj else "divergem"
    ax.set_title(f"(b) A imaginação é fiel: imaginado vs real ({match})")
    ax.legend(loc="upper right", fontsize=7.5)

    # (c) estatística sobre pares início→objetivo aleatórios.
    n_trials = 60
    p_succ = r_succ = 0
    p_steps, r_steps = [], []
    for _ in range(n_trials):
        free = world.free_states()
        st = free[rng.integers(len(free))]
        gl = free[rng.integers(len(free))]
        if st == gl:
            continue
        w = make_world(gl)
        m = WorldModel(); explore_and_learn(w, m, 3000, np.random.default_rng(int(rng.integers(1 << 30))))
        ok_p, sp = run_planner(w, m, st, gl)
        ok_r, sr = run_reactive(w, st, gl, rng)
        p_succ += ok_p; r_succ += ok_r
        if ok_p:
            p_steps.append(sp)
        if ok_r:
            r_steps.append(sr)
    ax = axes[1, 0]
    bars = ax.bar(["planejador", "reativo"], [100 * p_succ / n_trials, 100 * r_succ / n_trials],
                  color=["#2ca02c", "#d62728"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"(c) Taxa de sucesso ({n_trials} pares início→objetivo)")
    ax.set_ylabel("% que chega ao objetivo"); ax.set_ylim(0, 105)

    # (d) resumo.
    ax = axes[1, 1]; ax.axis("off")
    pm = np.mean(p_steps) if p_steps else float("nan")
    rm = np.mean(r_steps) if r_steps else float("nan")
    ax.text(0.0, 0.98,
            "M11 — agência: planejar = \"pensar\"\n\n"
            "O agente APRENDE o modelo (o que cada ação faz)\n"
            "vivendo, depois SIMULA ações na imaginação para\n"
            "achar o caminho — sem tentar no mundo real.\n\n"
            f"• sucesso: planejador {100*p_succ/n_trials:.0f}% vs reativo {100*r_succ/n_trials:.0f}%\n"
            f"• passos médios (quando chega): plan {pm:.0f} / react {rm:.0f}\n"
            f"• imaginação vs realidade: {match}\n\n"
            "\"Pensar = rodar a previsão por dentro.\" A mesma\n"
            "máquina de previsão, agora usada para AÇÃO.\n"
            "Ponte direta para a cognição emergente (Fase 3).",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M11 — Agência: o agente imagina futuros e planeja (o primeiro \"pensar\")",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m11_planning.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"Sucesso: planejador {100*p_succ/n_trials:.0f}%  reativo {100*r_succ/n_trials:.0f}%")
    print(f"Passos médios: planejador {pm:.1f}  reativo {rm:.1f}")
    print(f"Imaginação vs realidade (exemplo): {match}")


if __name__ == "__main__":
    main()
