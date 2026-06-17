"""M18 — Comunicação ANCORADA na percepção (Fase 3).

Os símbolos deixam de referir ids dados e passam a referir CATEGORIAS PERCEPTUAIS
(Harnad): o emissor percebe um objeto, categoriza-o, e nomeia; o receptor decodifica
e identifica o objeto. O léxico gruda na percepção de cada agente (à la language
games de Steels).

Demonstra:
  (a) com percepções alinhadas, a comunicação ANCORADA funciona (acaso → 100%);
  (b) o achado honesto: a comunicação é LIMITADA pelo alinhamento perceptual —
      só se comunica o que ambos conseguem DISTINGUIR (comm ≈ discriminabilidade);
  (c) o léxico ancorado: cada objeto percebido → seu símbolo.

Uso:   python experiments/m18_grounded.py
Salva: experiments/output/m18_grounded.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import GroundedLanguageGame  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIDE = 4
D = SIDE * SIDE
K = 6
N_ROUNDS = 5000


def objects():
    """K objetos distintos (4 barras + 2 diagonais), 4x4."""
    pats = []
    for r in range(4):
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    pats.append(np.eye(SIDE).ravel())
    pats.append(np.fliplr(np.eye(SIDE)).ravel())
    P = np.array(pats[:K])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def run(P, noise, seed):
    g = GroundedLanguageGame(P, recv_noise=noise, lr=0.15, temp=0.4, seed=seed)
    rng = np.random.default_rng(seed)
    curve_x, curve = [], []
    for t in range(N_ROUNDS):
        g.play(int(rng.integers(K)))
        if t % 100 == 0:
            curve_x.append(t); curve.append(100 * g.accuracy())
    return g, np.array(curve_x), np.array(curve)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    P = objects()

    # (a) caso alinhado (sem ruído perceptual).
    g_aligned, cx, ccurve = run(P, noise=0.0, seed=SEED)

    # (b) varre o ruído perceptual; coleta (discriminabilidade, comunicação).
    discs, comms = [], []
    for noise in np.linspace(0.0, 2.0, 11):
        for s in range(5):
            g, _, _ = run(P, noise=noise, seed=s)
            discs.append(100 * g.discriminability()); comms.append(100 * g.accuracy())
    discs = np.array(discs); comms = np.array(comms)
    corr = np.corrcoef(discs, comms)[0, 1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) curva de aprendizado (alinhado).
    ax = axes[0, 0]
    ax.plot(cx, ccurve, color="#2ca02c", lw=2)
    ax.axhline(100 / K, ls=":", color="#888", label=f"acaso ({100//K}%)")
    ax.set_title("(a) Comunicação ancorada funciona (percepções alinhadas)")
    ax.set_xlabel("rodadas"); ax.set_ylabel("acurácia (%)"); ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8)

    # (b) o achado: comunicação limitada pelo alinhamento perceptual.
    ax = axes[0, 1]
    ax.scatter(discs, comms, s=22, color="#1f77b4", alpha=0.6)
    ax.plot([60, 100], [60, 100], ls="--", color="#d62728", label="limite perceptual (y=x)")
    ax.set_title(f"(b) Só se comunica o que ambos DISTINGUEM (r={corr:.2f})")
    ax.set_xlabel("discriminabilidade perceptual do receptor (%)")
    ax.set_ylabel("acurácia de comunicação (%)")
    ax.legend(loc="upper left", fontsize=8)

    # (c) léxico ancorado: objeto percebido -> símbolo.
    ax = axes[1, 0]; ax.axis("off")
    ax.set_title("(c) Léxico ancorado: objeto percebido → símbolo")
    lex = [int(np.argmax(g_aligned.S[i])) for i in range(K)]
    for i in range(K):
        axc = fig.add_axes([0.08 + i * 0.066, 0.30, 0.055, 0.11])
        axc.imshow(P[i].reshape(SIDE, SIDE), cmap="magma"); axc.axis("off")
        fig.text(0.08 + i * 0.066 + 0.027, 0.27, f"→ {lex[i]}", ha="center",
                 fontsize=11, fontweight="bold")

    # (d) resumo.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M18 — comunicação ANCORADA na percepção\n\n"
            "Símbolo refere uma CATEGORIA PERCEPTUAL (Harnad),\n"
            "não um id dado. O emissor percebe→categoriza→nomeia;\n"
            "o receptor decodifica→identifica o objeto.\n\n"
            f"• alinhado: comunicação {ccurve[-1]:.0f}%  (acaso {100//K}%)\n"
            f"• comm × discriminabilidade: correlação r = {corr:.2f}\n"
            "  (a comunicação NÃO ultrapassa o que o receptor distingue)\n\n"
            "Achado honesto: grounding de verdade depende de conceitos\n"
            "COMPATÍVEIS. Se um agente não distingue dois objetos, não\n"
            "há símbolo que os separe — a percepção limita a linguagem.\n\n"
            "Honesto: escala de brinquedo, não linguagem plena. Mas é o\n"
            "primeiro grounding real — símbolos que significam o que o\n"
            "agente aprendeu a perceber.",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M18 — Símbolos ancorados na percepção: comunicar é limitado pelo que se distingue",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m18_grounded.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"Alinhado: comunicação final {ccurve[-1]:.0f}%  (acaso {100//K}%)")
    print(f"Correlação comm × discriminabilidade perceptual: r = {corr:.2f}")
    print(f"Léxico ancorado (objeto->símbolo): {lex}")


if __name__ == "__main__":
    main()
