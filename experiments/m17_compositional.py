"""M17 — Proto-sintaxe: composicionalidade (Fase 3).

De "palavras isoladas" (M16) para "frases com estrutura". Os conceitos agora são
COMPOSTOS (atributo1 × atributo2) e a mensagem tem 2 símbolos. A pergunta de ouro:
o código emergente GENERALIZA para combinações nunca vistas?

  • composicional (símbolo por atributo): pode generalizar — a produtividade da
    linguagem (infinitos significados de poucos símbolos).
  • holístico (mensagem inteira decorada): memoriza o treino, não generaliza.

Demonstra que a ESTRUTURA composicional é o que dá generalização — a essência da
sintaxe. Honesto: damos aos agentes a CAPACIDADE de compor; o que emerge é o código
consistente e sua generalização.

Uso:   python experiments/m17_compositional.py
Salva: experiments/output/m17_compositional.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import CompositionalGame  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
K1, K2 = 3, 3
N_SYMBOLS = 4
N_ROUNDS = 6000

ALL = [(a, b) for a in range(K1) for b in range(K2)]
HELDOUT = [(0, 0), (1, 1), (2, 2)]          # nunca vistas no treino
TRAIN = [c for c in ALL if c not in HELDOUT]


def train_game(mode, seed=SEED, track=False):
    g = CompositionalGame(K1, K2, N_SYMBOLS, mode=mode, lr=0.15, temp=0.4, seed=seed)
    rng = np.random.default_rng(seed)
    curve_x, curve = [], []
    for t in range(N_ROUNDS):
        a1, a2 = TRAIN[rng.integers(len(TRAIN))]
        g.play(a1, a2)
        if track and t % 100 == 0:
            curve_x.append(t); curve.append(100 * g.accuracy(TRAIN))
    return g, np.array(curve_x), np.array(curve)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    comp, cx, ccurve = train_game("compositional", track=True)
    holi, _, _ = train_game("holistic")

    # média sobre seeds para a barra de generalização (robustez)
    def gen_stats(mode):
        tr, ho = [], []
        for s in range(6):
            g, _, _ = train_game(mode, seed=s)
            tr.append(g.accuracy(TRAIN)); ho.append(g.accuracy(HELDOUT))
        return 100 * np.mean(tr), 100 * np.mean(ho)
    comp_tr, comp_ho = gen_stats("compositional")
    holi_tr, holi_ho = gen_stats("holistic")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) acurácia no treino sobe.
    ax = axes[0, 0]
    ax.plot(cx, ccurve, color="#2ca02c", lw=2)
    ax.set_title("(a) Aprende a comunicar conceitos compostos")
    ax.set_xlabel("rodadas"); ax.set_ylabel("acurácia no treino (%)"); ax.set_ylim(0, 105)

    # (b) O RESULTADO: generalização para combinações nunca vistas.
    ax = axes[0, 1]
    x = np.arange(2); w = 0.35
    ax.bar(x - w / 2, [comp_tr, holi_tr], w, label="treino (visto)", color="#888")
    ax.bar(x + w / 2, [comp_ho, holi_ho], w, label="HELD-OUT (nunca visto)",
           color=["#2ca02c", "#d62728"])
    for i, v in enumerate([comp_ho, holi_ho]):
        ax.text(i + w / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["composicional", "holístico"])
    ax.set_ylabel("acurácia (%)"); ax.set_ylim(0, 110)
    ax.set_title("(b) Composicional GENERALIZA; holístico só decora")
    ax.legend(loc="upper right", fontsize=8)

    # (c) o código composicional emergente: atributo -> símbolo.
    ax = axes[1, 0]; ax.axis("off")
    sym1 = [int(np.argmax(comp.S1[a])) for a in range(K1)]
    sym2 = [int(np.argmax(comp.S2[a])) for a in range(K2)]
    txt = "(c) Código composicional emergente\n\n"
    txt += "atributo 1 (posição 1 da mensagem):\n"
    for a in range(K1):
        txt += f"    valor {a}  →  símbolo {sym1[a]}\n"
    txt += "\natributo 2 (posição 2 da mensagem):\n"
    for a in range(K2):
        txt += f"    valor {a}  →  símbolo {sym2[a]}\n"
    txt += "\nmensagem p/ composto (a1,a2) = [sím(a1), sím(a2)]\n"
    txt += "símbolos = ATRIBUTOS => combinações novas decodificam."
    ax.text(0.0, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9.5, family="monospace")

    # (d) resumo.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M17 — proto-sintaxe / composicionalidade\n\n"
            f"Conceitos compostos: {K1}×{K2}={len(ALL)}; treino em {len(TRAIN)},\n"
            f"teste em {len(HELDOUT)} combinações NUNCA vistas.\n\n"
            f"Generalização (held-out, média de 6 seeds):\n"
            f"  • composicional: {comp_ho:.0f}%   ← produtividade!\n"
            f"  • holístico:     {holi_ho:.0f}%   ← só decorou\n\n"
            "A estrutura composicional é o que dá GENERALIZAÇÃO:\n"
            "símbolos viram atributos, então combinações novas\n"
            "se decodificam sozinhas. É a essência da sintaxe —\n"
            "infinitos significados de poucos símbolos.\n\n"
            "Honesto: damos a CAPACIDADE de compor; emergem o\n"
            "código consistente e a generalização. Linguagem\n"
            "plena (recursão, semântica) segue no horizonte.",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M17 — Proto-sintaxe: o código composicional generaliza para o que nunca viu",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m17_compositional.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"Composicional: treino {comp_tr:.0f}%  held-out {comp_ho:.0f}%")
    print(f"Holístico:     treino {holi_tr:.0f}%  held-out {holi_ho:.0f}%")
    print(f"Código: atributo1 {sym1}  atributo2 {sym2}")


if __name__ == "__main__":
    main()
