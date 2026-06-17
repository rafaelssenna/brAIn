"""M10 — Predictive coding SPIKING (Fase 2, o salto purista).

O sistema da H1 finalmente num substrato unificado: as causas latentes são
NEURÔNIOS LIF (M1) reais; a predição é aprendida (M4) por uma regra LOCAL sobre a
taxa de disparo. Sem backprop, sem rate-units contínuas escondidas.

Demonstra:
  (a) a rede SPIKING aprende: a surpresa cai com a experiência;
  (b) é genuinamente spiking: raster mostra padrões diferentes ativando
      neurônios diferentes (códigos específicos por padrão);
  (c) campos receptivos emergem (colunas de W) recuperando os padrões do mundo.

Honesto: é *rate-coded* (erro lido em taxa), pequeno (D=16) e lento (laço spiking
em Python). É prova de conceito do substrato — integrar no corpo/curiosidade e
escalar fica para depois (M11/M12).

Uso:   python experiments/m10_spiking.py
Salva: experiments/output/m10_spiking.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import SpikingPredictiveCoder  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIDE = 4
D = SIDE * SIDE
N_LATENT = 6
N_STEPS = 3000


def world_patterns():
    """4 barras horizontais distintas num grid 4x4."""
    pats = []
    for r in range(SIDE):
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    P = np.array(pats)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    P = world_patterns()
    K = len(P)
    spc = SpikingPredictiveCoder(n_obs=D, n_latent=N_LATENT, seed=0)
    rng = np.random.default_rng(SEED)

    errs = []
    for _ in range(N_STEPS):
        x = P[rng.integers(K)] + 0.05 * rng.standard_normal(D)
        _, e = spc.learn(x)
        errs.append(e)
    win = 50
    smooth = np.convolve(errs, np.ones(win) / win, mode="valid")
    drop = 100 * (1 - smooth[-1] / smooth[0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (a) surpresa caindo.
    ax = axes[0, 0]
    ax.plot(smooth, color="#1f77b4", lw=1.8)
    ax.set_title("(a) A rede SPIKING aprende a prever (surpresa cai)")
    ax.set_xlabel("experiência (passo)"); ax.set_ylabel("surpresa ‖ε‖² (média móvel)")
    ax.text(0.96, 0.95, f"{smooth[0]:.2f} → {smooth[-1]:.2f}  (−{drop:.0f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)

    # (b) raster: dois padrões ativam neurônios diferentes.
    ax = axes[0, 1]
    for pi, color in [(0, "#2ca02c"), (1, "#d62728")]:
        times = spc.spike_raster(P[pi])
        for ni, ts in enumerate(times):
            y = ni + (0.0 if pi == 0 else 0.35)
            ax.scatter(ts, np.full_like(ts, y), s=10, color=color)
    ax.plot([], [], "o", color="#2ca02c", label="padrão 0")
    ax.plot([], [], "o", color="#d62728", label="padrão 1")
    ax.set_title("(b) É spiking: padrões diferentes, neurônios diferentes")
    ax.set_xlabel("tempo (ms)"); ax.set_ylabel("neurônio latente")
    ax.set_yticks(range(N_LATENT)); ax.legend(loc="upper right", fontsize=8)

    # (c) campos receptivos aprendidos vs padrões reais.
    ax = axes[1, 0]
    canvas = _tile(spc.W, SIDE, cols=N_LATENT)
    ax.imshow(canvas, cmap="magma")
    ax.set_title("(c) Campos receptivos aprendidos (colunas de W)")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M10 — predictive coding SPIKING (substrato unificado)\n\n"
            "Unidades latentes = NEURÔNIOS LIF (M1).\n"
            "Predição aprendida (M4) por regra LOCAL: ΔW ∝ ε·ρᵀ\n"
            "sobre a TAXA DE DISPARO. Sem backprop.\n\n"
            f"• mundo: {K} padrões (barras 4x4) + ruído\n"
            f"• surpresa: {smooth[0]:.2f} → {smooth[-1]:.2f}  (−{drop:.0f}%)\n"
            "• não-negatividade vem de graça (limiar do spike)\n\n"
            "A fratura desde o M4 (rate vs spiking) está costurada:\n"
            "spiking + predição + plasticidade local numa rede só.\n\n"
            "Honesto: rate-coded (erro em taxa), pequeno e lento.\n"
            "Integrar ao corpo/curiosidade e escalar = M11/M12.",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M10 — Predictive coding em neurônios spiking: o substrato da H1, unificado",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m10_spiking.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"Surpresa: {smooth[0]:.3f} -> {smooth[-1]:.3f}  (queda {drop:.0f}%)")


def _tile(W, side, cols):
    n = W.shape[1]
    canvas = np.full((side, cols * (side + 1) - 1), np.nan)
    for i in range(n):
        canvas[:, i * (side + 1):i * (side + 1) + side] = W[:, i].reshape(side, side)
    return canvas


if __name__ == "__main__":
    main()
