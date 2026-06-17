"""M6 — Estudo de ablação (cada peça justifica sua existência?).

Regra do roadmap nº3: toda peça precisa justificar por que existe. Aqui tiramos
um componente de cada vez do predictive coder (M4) e medimos o estrago no erro
de previsão held-out. Separa o que é LOAD-BEARING do que é refinamento.

Ablações:
  • completo          — modelo cheio (referência)
  • sem aprendizado   — eta_w = 0 (W fica aleatório: nunca aprende)
  • sem inferência    — n_infer = 1 (r mal sai de zero: não "pensa" sobre a entrada)
  • poucos latentes   — n_latent = 4  (< 8 padrões do mundo: capacidade insuficiente)
  • sem não-negativ.  — nonneg = False
  • sem prior L2      — l2_prior = 0

Uso:   python experiments/m6_ablations.py
Salva: experiments/output/m6_ablations.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import PredictiveCoder  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GRID = 8
D = GRID * GRID
NOISE = 0.06
N_STEPS = 6000


def world_patterns():
    pats = []
    for r in range(4):
        p = np.zeros((GRID, GRID)); p[r * 2, :] = 1.0; pats.append(p)
    for c in range(4):
        p = np.zeros((GRID, GRID)); p[:, c * 2] = 1.0; pats.append(p)
    P = np.array([p.ravel() for p in pats])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


P = world_patterns()
K = len(P)

BASE = dict(n_latent=12, n_infer=30, eta_w=0.05, l2_prior=0.1, nonneg=True)
ABLATIONS = {
    "completo":          {},
    "sem aprendizado":   {"eta_w": 0.0},
    "sem inferência":    {"n_infer": 1},
    "poucos latentes":   {"n_latent": 4},
    "sem não-negativ.":  {"nonneg": False},
    "sem prior L2":      {"l2_prior": 0.0},
}


def train_eval(overrides, n_steps=N_STEPS, seed=0):
    cfg = dict(BASE); cfg.update(overrides)
    pc = PredictiveCoder(n_obs=D, eta_r=0.1, seed=seed, **cfg)
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_steps):
        x = P[rng.integers(K)] + NOISE * rng.standard_normal(D)
        _, e = pc.learn(x)
        errs.append(e)
    held = float(np.mean([pc.prediction_error(P[rng.integers(K)] + NOISE * rng.standard_normal(D))
                          for _ in range(3000)]))
    return np.array(errs), held


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results, curves = {}, {}
    for name, ov in ABLATIONS.items():
        errs, held = train_eval(ov, seed=SEED)
        results[name] = held
        win = 100
        curves[name] = np.convolve(errs, np.ones(win) / win, mode="valid")
        print(f"  {name:18s} -> surpresa held-out = {held:.3f}")

    rank = int(np.linalg.matrix_rank(P))
    floor = NOISE ** 2 * (D - rank)        # ótimo de subespaço
    base_zero = 1.0 + NOISE ** 2 * D       # prever nada

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # (a) curvas de surpresa para as ablações mais ilustrativas.
    for name, color in [("completo", "#2ca02c"), ("sem inferência", "#ff7f0e"),
                        ("sem aprendizado", "#d62728")]:
        ax1.plot(curves[name], lw=1.8, color=color, label=name)
    ax1.axhline(floor, ls="-", lw=1, color="#888", label=f"ótimo de subespaço ({floor:.2f})")
    ax1.set_title("(a) Tirar inferência ou aprendizado mata a previsão")
    ax1.set_xlabel("experiência (passo)"); ax1.set_ylabel("surpresa ‖ε‖² (média móvel)")
    ax1.legend(loc="upper right", fontsize=8); ax1.set_ylim(0, base_zero * 1.05)

    # (b) surpresa final de cada ablação.
    names = list(results.keys())
    vals = [results[n] for n in names]
    colors = ["#2ca02c"] + ["#1f77b4"] * (len(names) - 1)
    bars = ax2.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax2.axvline(floor, ls="-", lw=1.2, color="#888")
    ax2.text(floor, -0.6, f"ótimo\n{floor:.2f}", fontsize=7, color="#555", ha="center")
    for b, v in zip(bars, vals[::-1]):
        ax2.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                 va="center", fontsize=8)
    ax2.set_title("(b) Surpresa final por ablação (menor = melhor)")
    ax2.set_xlabel("surpresa held-out ‖ε‖²")

    fig.suptitle("brAIn · M6 — Ablação: o que é essencial vs. o que é refinamento",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT_DIR, "m6_ablations.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"Referências: ótimo de subespaço={floor:.3f}  prever-nada={base_zero:.3f}")


if __name__ == "__main__":
    main()
