"""M22 — Substrato esparso e dirigido a eventos: método > força bruta.

O cérebro roda com <20W porque é ESPARSO (poucos neurônios ativos por vez) e
DIRIGIDO A EVENTOS (só quem dispara custa). Aqui medimos esse princípio sobre a
máquina de previsão do M4: um código latente esparso (k-winners-take-all) atinge a
MESMA qualidade de previsão/discriminação que o denso, gastando MUITO menos operações
sinápticas (SynOps). A moeda é a CONTAGEM de operações (independente de hardware),
não o relógio — em Python o laço esparso é até mais lento (o M6 já mostrou), mas o
número de operações é o que hardware neuromórfico converte em energia.

O que isto PROVA: o princípio de eficiência em miniatura (esparsidade mantém a
qualidade cortando operações). O que NÃO prova: escala, wall-clock, ou eficiência
biológica (estamos a ~9 ordens de magnitude do cérebro).

Uso:   python experiments/m22_sparse_efficient.py
Salva: experiments/output/m22_sparse_efficient.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import (SparsePredictiveCoder, OpCounter, dense_learn_macs,  # noqa: E402
                   spiking_learn_ops)

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GRID = 8
D = GRID * GRID            # 64
N_LATENT = 16
NOISE = 0.06
N_STEPS = 8000
N_SWEEP = 5000
K_MAIN = 2                 # k-WTA principal: 2/16 = 12.5% ativo
N_INFER = 30
ETA_W = 0.05
# parâmetros do baseline spiking (M10) só para a fórmula de energia
SPK_CYCLES, SPK_WINDOW = 14, 50


def make_world_prototypes():
    """K=8 barras orientadas (4 horizontais + 4 verticais) num grid 8x8 (igual M4)."""
    protos = []
    for row in range(0, GRID, 2):
        p = np.zeros((GRID, GRID)); p[row, :] = 1.0; protos.append(p)
    for col in range(0, GRID, 2):
        p = np.zeros((GRID, GRID)); p[:, col] = 1.0; protos.append(p)
    P = np.array([p.ravel() for p in protos])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def train(coder, P, K, n_steps, rng, count=False):
    """Vive o mundo aprendendo a prever. Devolve a curva de surpresa e SynOps acumulado."""
    cnt = OpCounter() if count else None
    errs, cum = [], []
    for _ in range(n_steps):
        x = P[rng.integers(K)] + NOISE * rng.standard_normal(D)
        _, e = coder.learn(x, counter=cnt)
        errs.append(e)
        if count:
            cum.append(cnt.total_warm())
    return np.array(errs), (np.array(cum) if count else None)


def held_out(coder, P, K, rng, n=4000):
    return float(np.mean([coder.prediction_error(P[rng.integers(K)] +
                          NOISE * rng.standard_normal(D)) for _ in range(n)]))


def discriminability(coder, P):
    """Quantos dos K padrões o agente mapeia para conceitos (argmax de r) distintos."""
    cs = [int(np.argmax(coder.infer(P[o]))) for o in range(len(P))]
    return len(set(cs)) / len(P)


def mean_active_frac(coder, P):
    return float(np.mean([coder.active_fraction(P[o]) for o in range(len(P))]))


def synops_per_infer(coder, P):
    """SynOps (dirigido a eventos) de uma inferência média, num modelo já treinado."""
    cnt = OpCounter()
    for o in range(len(P)):
        coder.infer(P[o], counter=cnt)
    return cnt.total_warm() / len(P)


def synops_to_target(cum, errs, target, per_step_dense=None):
    """SynOps acumulado até a surpresa (média móvel) cruzar o alvo. Honra o M6:
    mede operações ATÉ a qualidade, não por passo."""
    win = 100
    sm = np.convolve(errs, np.ones(win) / win, mode="valid")
    hit = np.argmax(sm <= target) if np.any(sm <= target) else None
    if hit is None:
        return None
    if per_step_dense is not None:                  # baseline denso: fórmula fechada
        return per_step_dense * (hit + win)
    return float(cum[hit + win - 1])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    P = make_world_prototypes()
    K = len(P)

    common = dict(n_obs=D, n_latent=N_LATENT, n_infer=N_INFER, eta_w=ETA_W, seed=0)

    # --- (1) denso/nonneg (= M4) e (2) k-WTA (= M22), mesma semente/mundo ---
    dense = SparsePredictiveCoder(**common, k_active=None, l1=0.0)     # nonneg-only
    sparse = SparsePredictiveCoder(**common, k_active=K_MAIN, l1=0.0)  # k-WTA

    e_dense, cum_dense = train(dense, P, K, N_STEPS, np.random.default_rng(1), count=True)
    e_sparse, cum_sparse = train(sparse, P, K, N_STEPS, np.random.default_rng(1), count=True)

    win = 100
    sm_dense = np.convolve(e_dense, np.ones(win) / win, mode="valid")
    sm_sparse = np.convolve(e_sparse, np.ones(win) / win, mode="valid")

    held_dense = held_out(dense, P, K, np.random.default_rng(7))
    held_sparse = held_out(sparse, P, K, np.random.default_rng(7))
    disc_dense = discriminability(dense, P)
    disc_sparse = discriminability(sparse, P)
    af_dense = mean_active_frac(dense, P)
    af_sparse = mean_active_frac(sparse, P)

    # baselines / âncoras (metodologia do M4)
    sigma2 = NOISE ** 2
    rank = int(np.linalg.matrix_rank(P))
    floor_sub = sigma2 * (D - rank)
    base_zero = float(np.mean([np.sum((P[rng.integers(K)] +
                       NOISE * rng.standard_normal(D)) ** 2) for _ in range(2000)]))

    # --- energia (SynOps por inferência) ---
    syn_dense_full = dense_learn_macs(D, N_LATENT, N_INFER)            # denso pleno (todo neurônio)
    syn_nonneg = synops_per_infer(dense, P)                           # nonneg só (conta as ativas)
    syn_sparse = synops_per_infer(sparse, P)                          # k-WTA
    syn_spiking = spiking_learn_ops(D, N_LATENT, SPK_CYCLES, SPK_WINDOW)  # M10 (com passos LIF)
    factor = syn_dense_full / syn_sparse

    # --- SynOps ATÉ a qualidade (honra o M6: não é por passo) ---
    target = floor_sub + 0.20 * (base_zero - floor_sub)               # alvo comum alcançável
    to_dense = synops_to_target(None, e_dense, target, per_step_dense=syn_dense_full)
    to_sparse = synops_to_target(cum_sparse, e_sparse, target)
    ratio_tt = (to_dense / to_sparse) if (to_dense and to_sparse) else None

    # --- (3) fronteira de Pareto: varre k, mede esparsidade x qualidade x discriminação ---
    ks = [1, 2, 3, 4, 6, 8, 12, 16]
    pf_af, pf_held, pf_disc = [], [], []
    for k in ks:
        c = SparsePredictiveCoder(**common, k_active=(None if k >= N_LATENT else k), l1=0.0)
        train(c, P, K, N_SWEEP, np.random.default_rng(1))
        pf_af.append(100 * mean_active_frac(c, P))
        pf_held.append(held_out(c, P, K, np.random.default_rng(7)))
        pf_disc.append(discriminability(c, P))

    # ============================ figura ============================
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)

    # (a) surpresa: esparso acompanha o denso.
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(sm_dense, color="#1f77b4", lw=1.8, label=f"denso/nonneg (M4): {sm_dense[-1]:.3f}")
    ax.plot(sm_sparse, color="#2ca02c", lw=1.8, label=f"k-WTA k={K_MAIN} (M22): {sm_sparse[-1]:.3f}")
    ax.axhline(floor_sub, ls="-", lw=1.0, color="#888", label=f"ótimo de subespaço: {floor_sub:.3f}")
    ax.set_title("(a) Esparso mantém a qualidade (surpresa cai igual)")
    ax.set_xlabel("experiência (passo)"); ax.set_ylabel("erro de previsão ‖ε‖²")
    ax.set_ylim(0, base_zero * 0.6); ax.legend(loc="upper right", fontsize=7.5)

    # (b) ENERGIA — o painel-tese (SynOps por inferência, log).
    ax = fig.add_subplot(gs[0, 1])
    labels = ["denso\npleno", "nonneg\nsó (M4)", f"k-WTA\nk={K_MAIN} (M22)", "spiking\nM10"]
    vals = [syn_dense_full, syn_nonneg, syn_sparse, syn_spiking]
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_yscale("log"); ax.set_ylabel("operações sinápticas por inferência (SynOps)")
    ax.set_title(f"(b) Energia: k-WTA usa ~{factor:.0f}x menos SynOps que o denso pleno")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)

    # (c) fronteira de Pareto: esparsidade x qualidade x discriminação.
    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(pf_af, pf_held, c=pf_disc, cmap="RdYlGn", vmin=0.5, vmax=1.0,
                    s=90, edgecolor="k", linewidth=0.6, zorder=5)
    for k, xa, ya in zip(ks, pf_af, pf_held):
        ax.annotate(f"k={k}", (xa, ya), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(floor_sub, ls="-", lw=0.8, color="#888")
    ax.set_xlabel("% latentes ativas (esparsidade)"); ax.set_ylabel("surpresa held-out")
    ax.set_title("(c) Pareto: cor = discriminação (verde=boa)")
    plt.colorbar(sc, ax=ax, fraction=0.046, label="conceitos distinguidos")

    # (d) campos receptivos do k-WTA (estrutura preservada).
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(_tile(sparse.W, GRID), cmap="magma")
    ax.set_title("(d) Campos receptivos (k-WTA): estrutura preservada")
    ax.set_xticks([]); ax.set_yticks([])

    # (e) histograma de ativação: ~k/N ativo.
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(["denso/nonneg", f"k-WTA k={K_MAIN}"], [100 * af_dense, 100 * af_sparse],
           color=["#1f77b4", "#2ca02c"])
    ax.axhline(100 * K_MAIN / N_LATENT, ls="--", color="#888",
               label=f"orçamento k/N = {100*K_MAIN/N_LATENT:.0f}%")
    ax.set_ylabel("% latentes ativas"); ax.set_title("(e) Esparsidade alcançada")
    ax.legend(fontsize=8)

    # (f) resumo honesto.
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    tt = f"{ratio_tt:.1f}x" if ratio_tt else "n/d"
    ax.text(0.0, 0.99,
            "M22 — substrato esparso (método > força bruta)\n\n"
            "O cérebro: <20W porque é ESPARSO + dirigido a\n"
            "eventos. Aqui, sobre a máquina de previsão (M4):\n\n"
            f"• qualidade: denso {held_dense:.3f} vs esparso {held_sparse:.3f}\n"
            f"  (held-out; esparso mantém)\n"
            f"• conceitos: denso {disc_dense*100:.0f}% vs esparso {disc_sparse*100:.0f}% distinguidos\n"
            f"• atividade: {af_dense*100:.0f}% -> {af_sparse*100:.0f}% (cortical ~5-10%)\n"
            f"• ENERGIA: ~{factor:.0f}x menos SynOps por inferência\n"
            f"• SynOps ATÉ a qualidade: ~{tt} menos (honra o M6)\n\n"
            "Honesto: a moeda é OPERAÇÃO SINÁPTICA, não relógio\n"
            "(em Python o esparso é mais lento). Prova o PRINCÍPIO\n"
            "de eficiência em miniatura; NÃO prova escala nem\n"
            "eficiência biológica (~9 ordens do cérebro).",
            transform=ax.transAxes, va="top", fontsize=8.4, family="monospace")

    fig.suptitle("brAIn · M22 — Eficiência pelo método: código esparso faz o mesmo com muito menos operações",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m22_sparse_efficient.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Qualidade held-out:  denso={held_dense:.4f}  esparso={held_sparse:.4f}")
    print(f"Discriminacao:       denso={disc_dense*100:.0f}%  esparso={disc_sparse*100:.0f}%")
    print(f"Atividade (ativo%):  denso={af_dense*100:.1f}%  esparso={af_sparse*100:.1f}%  (k/N={100*K_MAIN/N_LATENT:.0f}%)")
    print(f"SynOps/inferencia:   denso_pleno={syn_dense_full:.0f}  nonneg={syn_nonneg:.0f}  "
          f"k-WTA={syn_sparse:.0f}  spiking-M10={syn_spiking:.0f}")
    print(f"Fator de energia (denso_pleno / k-WTA): {factor:.1f}x")
    print(f"SynOps ate a qualidade (alvo={target:.3f}):  denso={to_dense}  esparso={to_sparse}  razao={tt}")
    print("Pareto (k, ativo%, held-out, discrimina):")
    for k, a, h, d in zip(ks, pf_af, pf_held, pf_disc):
        print(f"   k={k:2d}  ativo={a:4.1f}%  held={h:.3f}  discrimina={d*100:.0f}%")


def _tile(W, grid, cols=8):
    n = W.shape[1]; rows = int(np.ceil(n / cols))
    canvas = np.full((rows * (grid + 1) - 1, cols * (grid + 1) - 1), np.nan)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[r * (grid + 1):r * (grid + 1) + grid,
               c * (grid + 1):c * (grid + 1) + grid] = W[:, i].reshape(grid, grid)
    return canvas


if __name__ == "__main__":
    main()
