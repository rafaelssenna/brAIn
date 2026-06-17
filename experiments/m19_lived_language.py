"""M19 — A língua nasce da vivência (Fase 3).

Diferente do M16-M18, aqui os conceitos NÃO são dados. Cada agente tem seu próprio
predictive coder (o do M4) e APRENDE sozinho a categorizar os objetos do mundo.
Só então os dois jogam o jogo de linguagem sobre esses conceitos vividos.

Demonstra:
  (a) quando os dois aprendem bem (distinguem tudo), a língua EMERGE de conceitos
      auto-aprendidos: comunicação vai a ~100%;
  (b) achado honesto: a comunicação acompanha o quanto o agente mais fraco
      consegue DISTINGUIR — só se fala do que ambos aprenderam a separar;
  (c) cada agente formou seus próprios conceitos (campos receptivos) vivendo o mundo.

Uso:   python experiments/m19_lived_language.py
Salva: experiments/output/m19_lived_language.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import PredictiveCoder, LivedLanguageGame  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIDE = 8
D = SIDE * SIDE
K = 8
NOISE = 0.06


def objects():
    pats = []
    for r in range(0, SIDE, 2):
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    for c in range(0, SIDE, 2):
        p = np.zeros((SIDE, SIDE)); p[:, c] = 1.0; pats.append(p.ravel())
    P = np.array(pats[:K])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def learn_concepts(P, n_latent, seed, steps=5000):
    """Um agente aprende a categorizar os objetos vivendo o mundo (predictive
    coder). Retorna (pc, mapa objeto->conceito aprendido)."""
    pc = PredictiveCoder(n_obs=D, n_latent=n_latent, eta_w=0.05, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        pc.learn(P[rng.integers(K)] + NOISE * rng.standard_normal(D))
    cmap = np.array([int(np.argmax(pc.infer(P[i]))) for i in range(K)])
    return pc, cmap


def communicate(cmapA, cmapB, seed, rounds=6000, track=False):
    g = LivedLanguageGame(cmapA, cmapB, n_symbols=K, seed=seed)
    rng = np.random.default_rng(seed)
    xs, curve = [], []
    for t in range(rounds):
        g.play(int(rng.integers(K)))
        if track and t % 100 == 0:
            xs.append(t); curve.append(100 * g.accuracy())
    return g, np.array(xs), np.array(curve)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    P = objects()

    # (a) caso ideal: dois agentes aprendem bem (capacidade de sobra)
    pcA, cmapA = learn_concepts(P, 12, seed=1)
    pcB, cmapB = learn_concepts(P, 12, seed=2)
    g0, cx, ccurve = communicate(cmapA, cmapB, seed=SEED, track=True)

    # (b) varia a capacidade perceptual do agente B (menos latentes = funde objetos)
    pts_disc, pts_acc = [], []
    for nl in [3, 4, 5, 6, 8, 12]:
        for s in range(2):
            _, cmB = learn_concepts(P, nl, seed=10 + s + nl)
            gg, _, _ = communicate(cmapA, cmB, seed=20 + s + nl)
            pts_disc.append(100 * gg.receiver_discrim()); pts_acc.append(100 * gg.accuracy())
    pts_disc = np.array(pts_disc); pts_acc = np.array(pts_acc)
    corr = np.corrcoef(pts_disc, pts_acc)[0, 1] if pts_disc.std() > 0 else float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(cx, ccurve, color="#2ca02c", lw=2)
    ax.axhline(100 / K, ls=":", color="#888", label=f"acaso ({100 // K}%)")
    ax.set_title("(a) Língua emerge de conceitos AUTO-APRENDIDOS (ambos distinguem tudo)")
    ax.set_xlabel("rodadas"); ax.set_ylabel("comunicação (%)"); ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[0, 1]
    ax.scatter(pts_disc, pts_acc, s=30, color="#1f77b4", alpha=0.7)
    ax.plot([30, 100], [30, 100], ls="--", color="#d62728", label="teto (y=x)")
    ax.set_title(f"(b) Comunicação x o que o agente B distingue (r={corr:.2f})")
    ax.set_xlabel("conceitos distintos que B aprendeu (%)")
    ax.set_ylabel("comunicação (%)"); ax.legend(loc="upper left", fontsize=8)

    ax = axes[1, 0]
    ax.imshow(_tile(pcA.W, SIDE, K), cmap="magma")
    ax.set_title("(c) Conceitos que o agente A formou vivendo (campos receptivos)")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M19 — a língua nasce da vivência\n\n"
            "Cada agente tem seu predictive coder e APRENDE sozinho\n"
            "a categorizar os objetos. Só então jogam o jogo de\n"
            "linguagem sobre esses conceitos vividos.\n\n"
            f"• ambos aprendendo bem: comunicação {ccurve[-1]:.0f}%\n"
            f"• comm x discriminabilidade de B: r = {corr:.2f}\n"
            "  (a língua não passa do que ambos distinguem)\n\n"
            "Pela primeira vez os símbolos significam conceitos que o\n"
            "próprio agente formou pela experiência, não etiquetas dadas.\n\n"
            "Honesto: escala de brinquedo; o alinhamento dos conceitos\n"
            "ainda depende de ambos aprenderem a separar as mesmas coisas.",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M19 — Língua ancorada em conceitos que cada agente aprendeu sozinho",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m19_lived_language.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"(a) ambos bem: comunicacao final {ccurve[-1]:.0f}%  (acaso {100 // K}%)")
    print(f"(b) comm x discriminabilidade de B: r = {corr:.2f}")
    print(f"    discrim de B: {np.round(pts_disc).astype(int)}")
    print(f"    comm:         {np.round(pts_acc).astype(int)}")


def _tile(W, side, cols):
    n = W.shape[1]
    rows = int(np.ceil(n / cols))
    canvas = np.full((rows * (side + 1) - 1, cols * (side + 1) - 1), np.nan)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[r * (side + 1):r * (side + 1) + side, c * (side + 1):c * (side + 1) + side] = W[:, i].reshape(side, side)
    return canvas


if __name__ == "__main__":
    main()
