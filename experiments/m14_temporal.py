"""M14 — Previsão temporal/sequencial (Fase 3, rumo à cognição).

O brAIn aprende a prever SEQUÊNCIAS no tempo. Demonstra:
  (a) APRENDE estrutura temporal: o erro cai numa sequência estruturada, mas
      NÃO numa sequência aleatória (não há o que prever) — controle.
  (b) IMAGINA a continuação: semeado com o início, gera o resto da sequência
      corretamente (pensar à frente no tempo).
  (c) O CONTEXTO importa: numa sequência em que o próximo depende do PASSADO
      (não só do presente), o modelo COM memória acerta; SEM memória, não.

Por que importa: prever sequências é o substrato de imaginar cadeias de futuros e
a ponte para a linguagem (predição sequencial sobre símbolos).

Uso:   python experiments/m14_temporal.py
Salva: experiments/output/m14_temporal.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import TemporalPredictiveCoder  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIDE = 4
D = SIDE * SIDE


def make_symbols(k, seed=1):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (k, D))
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def train_on(seq_idx, P, decay, n_reps, seed=SEED, noise=0.02):
    tpc = TemporalPredictiveCoder(D, decay=decay, seed=0)
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_reps):
        for i in seq_idx:
            x = P[i] + noise * rng.standard_normal(D)
            e, _ = tpc.observe(x)
            errs.append(e)
    return tpc, np.array(errs)


def smooth(e, w=50):
    return np.convolve(e, np.ones(w) / w, mode="valid")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- (a) estrutura vs aleatório ----
    P5 = make_symbols(5)
    melody = [0, 1, 2, 3, 4]
    _, err_struct = train_on(melody, P5, decay=0.5, n_reps=400)
    rng = np.random.default_rng(SEED)
    random_seq = list(rng.integers(5, size=5 * 400))
    tpc_r = TemporalPredictiveCoder(D, decay=0.5, seed=0)
    err_rand = []
    for i in random_seq:
        e, _ = tpc_r.observe(P5[i] + 0.02 * rng.standard_normal(D))
        err_rand.append(e)
    err_rand = np.array(err_rand)

    # ---- (b) geração / imaginação ----
    tpc_m, _ = train_on(melody, P5, decay=0.5, n_reps=500)
    tpc_m.reset_context()
    for i in [0, 1, 2]:                       # semeia com o início
        tpc_m.observe(P5[i], learn=False)
    generated = tpc_m.generate(8, P5)         # imagina o resto
    true_cont = [(3 + k) % 5 for k in range(8)]   # 3,4,0,1,2,3,4,0

    # ---- (c) contexto importa ----
    P4 = make_symbols(4, seed=2)
    ctx_seq = [0, 1, 0, 2]                     # após 0 vem 1 OU 2 (depende do passado)
    _, err_mem = train_on(ctx_seq, P4, decay=0.6, n_reps=500, seed=7)
    _, err_nomem = train_on(ctx_seq, P4, decay=0.0, n_reps=500, seed=7)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(smooth(err_struct), color="#2ca02c", lw=1.8, label="sequência estruturada")
    ax.plot(smooth(err_rand), color="#d62728", lw=1.5, label="sequência aleatória (controle)")
    ax.set_title("(a) Aprende a estrutura temporal — só quando existe")
    ax.set_xlabel("passo"); ax.set_ylabel("erro de previsão (média móvel)")
    ax.legend(loc="upper right", fontsize=9)

    # (b) imaginação: thumbnails gerados vs verdadeiros
    ax = axes[0, 1]; ax.axis("off")
    ax.set_title("(b) Imagina a continuação (semeado com 0,1,2)")
    n = len(generated)
    ok = generated == true_cont
    for k in range(n):
        axg = fig.add_axes([0.55 + k * 0.052, 0.74, 0.045, 0.09])
        axg.imshow(P5[generated[k]].reshape(SIDE, SIDE), cmap="magma"); axg.axis("off")
        axt = fig.add_axes([0.55 + k * 0.052, 0.60, 0.045, 0.09])
        axt.imshow(P5[true_cont[k]].reshape(SIDE, SIDE), cmap="magma"); axt.axis("off")
    fig.text(0.54, 0.79, "gerado", ha="right", va="center", fontsize=9)
    fig.text(0.54, 0.645, "real", ha="right", va="center", fontsize=9)
    fig.text(0.78, 0.55, f"{'✓ idênticos' if ok else 'diferem'}  "
             f"(gerado {generated} vs real {true_cont})", ha="center", fontsize=8)

    ax = axes[1, 0]
    ax.plot(smooth(err_mem), color="#2ca02c", lw=1.8, label="COM memória (decay=0.6)")
    ax.plot(smooth(err_nomem), color="#ff7f0e", lw=1.6, ls="--", label="SEM memória (1ª ordem)")
    ax.set_title("(c) Contexto importa: prever o próximo exige o passado")
    ax.set_xlabel("passo"); ax.set_ylabel("erro de previsão (média móvel)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M14 — previsão temporal/sequencial (Fase 3)\n\n"
            "Contexto = traço com vazamento do passado recente.\n"
            "Previsão x̂ = W·contexto; aprende ΔW ∝ erro·contextoᵀ\n"
            "(local, sem backprop).\n\n"
            f"(a) estrutura: erro final {smooth(err_struct)[-1]:.3f}\n"
            f"    aleatório:  erro final {smooth(err_rand)[-1]:.3f}\n"
            f"(b) imaginação da sequência: "
            f"{'correta' if ok else 'incorreta'}\n"
            f"(c) ctx-dependente: com mem {smooth(err_mem)[-1]:.3f}"
            f" vs sem mem {smooth(err_nomem)[-1]:.3f}\n\n"
            "Prever sequências = imaginar cadeias de futuros e\n"
            "a ponte para a linguagem. Primeiro passo da cognição.",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M14 — Previsão temporal: aprende sequências e imagina o que vem depois",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m14_temporal.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"(a) erro final estruturada={smooth(err_struct)[-1]:.3f}  aleatória={smooth(err_rand)[-1]:.3f}")
    print(f"(b) imaginação: gerado={generated}  real={true_cont}  {'OK' if ok else 'FALHOU'}")
    print(f"(c) ctx-dependente: com memória={smooth(err_mem)[-1]:.3f}  sem memória={smooth(err_nomem)[-1]:.3f}")


if __name__ == "__main__":
    main()
