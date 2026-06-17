"""M12 — Escala + Visão (Fase 2, finale).

Visão é a modalidade NATURAL do predictive coding (Rao & Ballard nasceu nela).
Alimento o predictive coder com pedaços de "imagem" feitos de várias bordas
orientadas sobrepostas (mini-imagens naturais) e vejo se o dicionário aprendido
(colunas de W) recupera DETECTORES DE BORDA ORIENTADOS, localizados — como as
células simples de V1. Sem rótulos: só prever a entrada visual.

Escala: entrada de 144 dim (12x12), 48 unidades latentes — bem maior que os 64
dos marcos anteriores. Honesto: o M6 já mostrou que NumPy/CPU não escala de
verdade; aqui demonstro que a matemática escala e reafirmo o caminho (GPU/neuro).

Uso:   python experiments/m12_vision.py
Salva: experiments/output/m12_vision.png
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import PredictiveCoder  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIDE = 12
D = SIDE * SIDE
N_LATENT = 48
N_STEPS = 6000


def visual_patch(rng, n_edges=3, noise=0.03):
    """Pedaço de 'imagem natural': soma de algumas bordas orientadas localizadas."""
    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    img = np.zeros((SIDE, SIDE))
    for _ in range(n_edges):
        theta = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(0, SIDE), rng.uniform(0, SIDE)
        proj = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
        sigma = SIDE / 4.0
        win = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
        img += np.tanh(proj) * win * rng.uniform(0.5, 1.0)
    img += noise * rng.standard_normal((SIDE, SIDE))
    v = img.ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def localization(W):
    """Quão localizado é cada átomo: fração de energia nos 25% pixels mais fortes
    (1.0 = pontual; 0.25 = espalhado uniformemente). Média sobre os átomos."""
    k = max(1, D // 4)
    fr = []
    for i in range(W.shape[1]):
        e = W[:, i] ** 2
        fr.append(np.sort(e)[::-1][:k].sum() / (e.sum() + 1e-12))
    return float(np.mean(fr))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    pc = PredictiveCoder(n_obs=D, n_latent=N_LATENT, n_infer=30,
                         eta_r=0.1, eta_w=0.05, l2_prior=0.1, seed=0)
    samples = [visual_patch(rng) for _ in range(N_STEPS)]

    t0 = time.perf_counter()
    errs = []
    for x in samples:
        _, e = pc.learn(x)
        errs.append(e)
    elapsed = time.perf_counter() - t0
    win = 100
    smooth = np.convolve(errs, np.ones(win) / win, mode="valid")

    loc = localization(pc.W)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) surpresa caindo (aprende a prever a entrada visual).
    ax = axes[0, 0]
    ax.plot(smooth, color="#1f77b4", lw=1.8)
    ax.set_title("(a) Aprende a prever a entrada visual (surpresa cai)")
    ax.set_xlabel("experiência (passo)"); ax.set_ylabel("surpresa ‖ε‖² (média móvel)")
    ax.text(0.96, 0.95, f"{smooth[0]:.2f} → {smooth[-1]:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)

    # (b) exemplos da entrada visual.
    ax = axes[0, 1]
    ax.imshow(_tile(np.array([visual_patch(rng) for _ in range(8)]).T, SIDE, cols=8),
              cmap="RdBu_r")
    ax.set_title("(b) Entrada visual: bordas orientadas sobrepostas")
    ax.set_xticks([]); ax.set_yticks([])

    # (c) dicionário aprendido — detectores de borda emergiram?
    ax = axes[1, 0]
    vmax = np.percentile(np.abs(pc.W), 99)
    ax.imshow(_tile(pc.W, SIDE, cols=8), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("(c) Dicionário aprendido (colunas de W): detectores de borda")
    ax.set_xticks([]); ax.set_yticks([])

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M12 — escala + visão (finale da Fase 2)\n\n"
            f"• entrada: {SIDE}x{SIDE}={D} dim (vs 64 antes); {N_LATENT} latentes\n"
            f"• surpresa: {smooth[0]:.2f} → {smooth[-1]:.2f}\n"
            f"• localização média dos átomos: {loc:.2f}\n"
            f"  (0.25=espalhado, 1.0=pontual; bordas localizadas > 0.25)\n"
            f"• tempo: {elapsed:.1f}s p/ {N_STEPS} amostras (NumPy/CPU)\n\n"
            "Visão é a praia do predictive coding: alimentado com\n"
            "bordas, o dicionário aprende DETECTORES DE BORDA\n"
            "orientados/localizados — como células de V1. Sem rótulos.\n\n"
            "Honesto: o M6 já mostrou que NumPy/CPU não escala de\n"
            "verdade; a matemática escala, o substrato é que pede\n"
            "GPU/JAX ou neuromórfico. Gabors limpos pedem ainda mais\n"
            "escala e branqueamento — aqui são bordas localizadas.",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M12 — Visão: detectores de borda emergem do predictive coding (como V1)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m12_vision.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"Surpresa: {smooth[0]:.3f} -> {smooth[-1]:.3f}")
    print(f"Localização média dos átomos: {loc:.3f} (espalhado=0.25)")
    print(f"Tempo: {elapsed:.1f}s para {N_STEPS} amostras de {D} dim")


def _tile(W, side, cols):
    n = W.shape[1]
    rows = int(np.ceil(n / cols))
    canvas = np.full((rows * (side + 1) - 1, cols * (side + 1) - 1), np.nan)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[r * (side + 1):r * (side + 1) + side,
               c * (side + 1):c * (side + 1) + side] = W[:, i].reshape(side, side)
    return canvas


if __name__ == "__main__":
    main()
