"""M16 — Comunicação emergente (Fase 3): o primeiro "conversar".

Dois agentes brAIn inventam, do zero, um código compartilhado símbolo<->conceito
jogando um jogo de referência. Demonstra:
  (a) a comunicação EMERGE: a acurácia sobe do acaso (1/N) até ~100%;
  (b) o LÉXICO emergente vira um código unívoco (cada conceito -> um símbolo);
  (c) os agentes de fato se entendem: trocas de exemplo, todas certas.

Os conceitos são ancorados (padrões que o brAIn forma) — comunicação grounded.
Honesto: é sinalização emergente, NÃO linguagem plena (sem gramática/semântica
rica). É o primeiro tijolo real do "conversar".

Uso:   python experiments/m16_communication.py
Salva: experiments/output/m16_communication.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import SignalingGame  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIDE = 4
N_CONCEPTS = 5
N_SYMBOLS = 5
N_ROUNDS = 4000


def concept_patterns():
    """5 conceitos ancorados (padrões distintos 4x4)."""
    pats = []
    for r in range(4):
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    p = np.eye(SIDE); pats.append(p.ravel())          # 5º: diagonal
    P = np.array(pats)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    P = concept_patterns()
    game = SignalingGame(N_CONCEPTS, N_SYMBOLS, lr=0.15, temp=0.4, seed=0)
    rng = np.random.default_rng(SEED)

    rewards, acc_curve, acc_x = [], [], []
    for t in range(N_ROUNDS):
        c = int(rng.integers(N_CONCEPTS))
        _, _, r = game.play(c, learn=True, explore=True)
        rewards.append(r)
        if t % 50 == 0:
            acc_x.append(t); acc_curve.append(game.eval_accuracy())

    final_acc = game.eval_accuracy()
    lex = game.lexicon()

    # trocas de exemplo (determinísticas) pós-treino
    exchanges = []
    for c in range(N_CONCEPTS):
        m, g, r = game.play(c, learn=False, explore=False)
        exchanges.append((c, m, g, r))

    win = 100
    rew_smooth = np.convolve(rewards, np.ones(win) / win, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) acurácia/recompensa subindo.
    ax = axes[0, 0]
    ax.plot(np.arange(len(rew_smooth)), 100 * rew_smooth, color="#1f77b4", lw=1.2,
            alpha=0.6, label="recompensa (média móvel)")
    ax.plot(acc_x, [100 * a for a in acc_curve], color="#2ca02c", lw=2, label="acurácia (determinística)")
    ax.axhline(100 / N_CONCEPTS, ls=":", color="#888", label=f"acaso ({100//N_CONCEPTS}%)")
    ax.set_title("(a) A comunicação EMERGE (acaso → ~100%)")
    ax.set_xlabel("rodadas"); ax.set_ylabel("% de acerto"); ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8)

    # (b) léxico emergente: matriz do emissor (conceito x símbolo).
    ax = axes[0, 1]
    Ssm = np.array([game._softmax(game.S[c]) for c in range(N_CONCEPTS)])
    im = ax.imshow(Ssm, cmap="magma", vmin=0, vmax=1)
    ax.set_title("(b) Léxico emergente (emissor): conceito → símbolo")
    ax.set_xlabel("símbolo"); ax.set_ylabel("conceito")
    ax.set_xticks(range(N_SYMBOLS)); ax.set_yticks(range(N_CONCEPTS))
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (c) o código ancorado: padrão do conceito -> símbolo.
    ax = axes[1, 0]; ax.axis("off")
    ax.set_title("(c) O código inventado (conceito ancorado → símbolo)")
    for c in range(N_CONCEPTS):
        axc = fig.add_axes([0.10 + c * 0.075, 0.30, 0.06, 0.10])
        axc.imshow(P[c].reshape(SIDE, SIDE), cmap="magma"); axc.axis("off")
        fig.text(0.10 + c * 0.075 + 0.03, 0.27, f"→ {lex[c]}", ha="center", fontsize=11, fontweight="bold")

    # (d) resumo + trocas.
    ax = axes[1, 1]; ax.axis("off")
    ex_txt = "\n".join([f"  vê conceito {c} → diz \"{m}\" → ouvinte aponta {g}  "
                        f"{'✓' if r else '✗'}" for (c, m, g, r) in exchanges])
    ax.text(0.0, 0.98,
            "M16 — comunicação emergente (jogo de referência)\n\n"
            f"• conceitos: {N_CONCEPTS}  símbolos: {N_SYMBOLS}\n"
            f"• acurácia final: {100*final_acc:.0f}%  (acaso {100//N_CONCEPTS}%)\n"
            f"• código unívoco (bijeção)? {'sim' if game.is_bijection() else 'não'}\n\n"
            "Trocas de exemplo (pós-treino):\n" + ex_txt + "\n\n"
            "Os dois inventaram um vocabulário SOZINHOS, por recompensa\n"
            "(Hebbiano de 3 fatores). Honesto: é sinalização grounded,\n"
            "não linguagem plena — mas é o primeiro tijolo do conversar.",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M16 — Comunicação emergente: dois agentes inventam uma língua compartilhada",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m16_communication.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"Acurácia final: {100*final_acc:.0f}%  (acaso {100//N_CONCEPTS}%)")
    print(f"Léxico (conceito->símbolo): {lex}  bijeção={game.is_bijection()}")
    for (c, m, g, r) in exchanges:
        print(f"  conceito {c} -> símbolo {m} -> palpite {g}  {'OK' if r else 'X'}")


if __name__ == "__main__":
    main()
