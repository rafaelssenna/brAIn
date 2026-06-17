"""M4 — A máquina de previsão (predictive coding).

O brAIn aprende a PREVER o próprio mundo sensorial, do zero, com regra LOCAL
(sem backprop). Demonstra três coisas:

  (a) A "SURPRESA" CAI: o erro de previsão despenca ao longo da vida do agente —
      a primeira curva de aprendizado de dentro pra fora, sem recompensa externa.
  (b) ESTRUTURA EMERGE: os campos receptivos (colunas de W) — que nasceram
      aleatórios — se organizam e recuperam os padrões do mundo. Ninguém
      programou isso; emergiu de minimizar a surpresa.
  (c) ANTES vs DEPOIS: a previsão da rede para uma entrada, de lixo (no início)
      a quase perfeita (no fim).

O "mundo" emite, a cada instante, um de K padrões (barras orientadas 8x8) com
ruído. A rede nunca é "ensinada" qual é qual — só tenta prever o que vê.

Uso:   python experiments/m4_predictive.py
Salva: experiments/output/m4_predictive.png
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
GRID = 8                 # patches 8x8
D = GRID * GRID          # dimensão sensorial = 64
N_STEPS = 8000           # passos de "vida"
NOISE = 0.06             # ruído sensorial (std por pixel; sinal tem norma 1)


def make_world_prototypes():
    """K=8 barras orientadas (4 horizontais + 4 verticais) num grid 8x8."""
    protos = []
    for row in range(0, GRID, 2):           # 4 barras horizontais
        p = np.zeros((GRID, GRID)); p[row, :] = 1.0; protos.append(p)
    for col in range(0, GRID, 2):           # 4 barras verticais
        p = np.zeros((GRID, GRID)); p[:, col] = 1.0; protos.append(p)
    P = np.array([p.ravel() for p in protos])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    P = make_world_prototypes()             # (K, D)
    K = len(P)
    pc = PredictiveCoder(n_obs=D, n_latent=12, n_infer=30,
                         eta_r=0.1, eta_w=0.05, l2_prior=0.1, seed=0)

    # Guarda o estado inicial (W aleatório) e um exemplo de previsão "antes".
    W_init = pc.W.copy()
    x_demo = P[0] + NOISE * rng.standard_normal(D)
    pred_before = pc.predict(pc.infer(x_demo))

    # --- Vida do agente: vê padrões com ruído e tenta prevê-los. ---
    errors = []
    for _ in range(N_STEPS):
        x = P[rng.integers(K)] + NOISE * rng.standard_normal(D)
        _, err = pc.learn(x)
        errors.append(err)
    errors = np.array(errors)

    pred_after = pc.predict(pc.infer(x_demo))

    # Média móvel do erro para uma curva limpa.
    win = 100
    smooth = np.convolve(errors, np.ones(win) / win, mode="valid")

    # --- Baselines e avaliação HELD-OUT (âncoras para interpretar a curva) ---
    sigma2 = NOISE ** 2
    rank = int(np.linalg.matrix_rank(P))         # dimensão do subespaço dos padrões
    floor_subspace = sigma2 * (D - rank)         # melhor resíduo possível p/ modelo de subespaço
    oracle_clean = sigma2 * D                    # prever o protótipo LIMPO deixa todo o ruído
    # Baseline "não prever nada" (r=0) e erro held-out (W congelado, fluxo novo):
    n_eval = 4000
    base_zero, held = [], []
    for _ in range(n_eval):
        x = P[rng.integers(K)] + NOISE * rng.standard_normal(D)
        base_zero.append(float(np.sum(x ** 2)))   # erro com r=0 (prevê nada)
        held.append(pc.prediction_error(x))        # W congelado: não aprende
    base_zero = float(np.mean(base_zero))
    held_err = float(np.mean(held))

    # ---------------- Figura ----------------
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

    # (a) Curva da surpresa, ANCORADA em baselines.
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(np.arange(len(smooth)), smooth, color="#9467bd", lw=1.8,
            label="brAIn (aprendendo)", zorder=5)
    ax.axhline(base_zero, ls=":", lw=1.0, color="#d62728",
               label=f"não prever nada (r=0): {base_zero:.2f}")
    ax.axhline(oracle_clean, ls="--", lw=1.0, color="#888",
               label=f"oráculo (prevê protótipo limpo): {oracle_clean:.3f}")
    ax.axhline(floor_subspace, ls="-", lw=1.0, color="#2ca02c",
               label=f"ótimo de subespaço σ²(D−d): {floor_subspace:.3f}")
    ax.set_title("(a) A surpresa cai com a vida (com baselines)")
    ax.set_xlabel("experiência (passo)")
    ax.set_ylabel("erro de previsão  ‖ε‖²  (média móvel)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, base_zero * 1.05)

    # (b) Antes vs depois da previsão.
    ax = fig.add_subplot(gs[0, 1])
    trio = np.hstack([x_demo.reshape(GRID, GRID),
                      np.ones((GRID, 1)) * np.nan,
                      pred_before.reshape(GRID, GRID),
                      np.ones((GRID, 1)) * np.nan,
                      pred_after.reshape(GRID, GRID)])
    ax.imshow(trio, cmap="magma")
    ax.set_title("(b) Entrada | Previsão ANTES | Previsão DEPOIS")
    ax.set_xticks([]); ax.set_yticks([])

    # (c) Texto-resumo (honesto, com held-out e gap ao ótimo).
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    drop = 100 * (1 - smooth[-1] / smooth[0])
    ax.text(0.0, 0.98,
            "Predictive coding — Rao & Ballard\n"
            "(instância linear, 1 camada)\n\n"
            f"• Mundo: {K} padrões (barras 8x8) + ruído\n"
            f"• Rede: 64→12, nasce com W aleatório\n"
            f"• Regra: ΔW ∝ ε·rᵀ  (LOCAL, sem backprop)\n\n"
            f"• Surpresa: {smooth[0]:.2f} → {smooth[-1]:.3f}  (−{drop:.0f}%)\n"
            f"• Held-out (W congelado): {held_err:.3f}\n"
            f"  ≈ treino → é aprendizado, não decoreba\n"
            f"• Ótimo de subespaço: {floor_subspace:.3f}\n"
            f"  (chega perto, fica logo acima)\n\n"
            f"• Sem recompensa, sem rótulos:\n  só prever o que vê.",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    # (d) Campos receptivos INICIAIS (aleatórios).
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(_tile(W_init, GRID), cmap="magma")
    ax.set_title("(d) Campos receptivos no NASCIMENTO (aleatórios)")
    ax.set_xticks([]); ax.set_yticks([])

    # (e) Campos receptivos APRENDIDOS.
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(_tile(pc.W, GRID), cmap="magma")
    ax.set_title("(e) Campos receptivos APRENDIDOS (estrutura emergiu)")
    ax.set_xticks([]); ax.set_yticks([])

    # (f) Padrões verdadeiros do mundo (para comparar com (e)).
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(_tile(P.T, GRID), cmap="magma")
    ax.set_title("(f) Padrões reais do mundo (gabarito)")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("brAIn · M4 — A máquina de previsão: aprende a prever o mundo, sem backprop",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = os.path.join(OUT_DIR, "m4_predictive.png")
    fig.savefig(out_path, dpi=120)
    print(f"Figura salva em: {out_path}")
    print(f"Surpresa (treino): {smooth[0]:.3f} -> {smooth[-1]:.4f} (queda de {drop:.0f}%)")
    print(f"Held-out (W congelado): {held_err:.4f}  (~= treino => aprendizado real)")
    print(f"Baselines:  nada(r=0)={base_zero:.3f}  "
          f"oraculo-limpo={oracle_clean:.3f}  otimo-subespaco(sigma^2*(D-{rank}))={floor_subspace:.3f}")


def _tile(W, grid, cols=6):
    """Empacota as colunas de W (campos receptivos) num mosaico para visualizar."""
    n = W.shape[1]
    rows = int(np.ceil(n / cols))
    canvas = np.full((rows * (grid + 1) - 1, cols * (grid + 1) - 1), np.nan)
    for i in range(n):
        r, c = divmod(i, cols)
        patch = W[:, i].reshape(grid, grid)
        canvas[r * (grid + 1):r * (grid + 1) + grid,
               c * (grid + 1):c * (grid + 1) + grid] = patch
    return canvas


if __name__ == "__main__":
    main()
