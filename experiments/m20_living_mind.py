"""M20 — O organismo vivo: percebe, lembra, é curioso e fala, num laço só.

A costura da cognição (Fase 3). Dois agentes, ao mesmo tempo e vivendo, aprendem a
PERCEBER o mundo (formar conceitos, M4/M9), LEMBRAM (replay, M8), são CURIOSOS
(learning progress, M5) e aprendem a FALAR (M16/M19) sobre o que percebem. Tudo
online, interligado, não em etapas separadas.

A pergunta científica: dá para ancorar a linguagem em conceitos que ainda estão se
formando? O resultado esperado (e o achado): primeiro o agente precisa aprender a
VER (estabilizar conceitos); só então a COMUNICAÇÃO consegue grudar neles. As duas
coisas co-emergem no mesmo laço, com a linguagem atrás da percepção.

Uso:   python experiments/m20_living_mind.py
Salva: experiments/output/m20_living_mind.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LivingAgent, IntrinsicMotivation  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIDE = 8
D = SIDE * SIDE
K = 6
N_LATENT = 10
N_SYMBOLS = 8
N_STEPS = 6000
NOISE = 0.06


def objects():
    pats = []
    for r in range(0, SIDE, 2):
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    for c in range(0, SIDE, 3):
        p = np.zeros((SIDE, SIDE)); p[:, c] = 1.0; pats.append(p.ravel())
    P = np.array(pats[:K])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def comm_acc(spk, lis, P):
    """Acurácia de comunicação (determinística) numa direção."""
    K = len(P)
    lc = [lis.concept(P[j]) for j in range(K)]
    tot = 0.0
    for o in range(K):
        m = spk.speak(spk.concept(P[o]), explore=False)
        cr = lis.listen(m)
        cands = [j for j in range(K) if lc[j] == cr]
        if o in cands:
            tot += 1.0 / len(cands)
    return tot / K


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    P = objects()
    A = LivingAgent(D, N_LATENT, N_SYMBOLS, seed=1)
    B = LivingAgent(D, N_LATENT, N_SYMBOLS, seed=2)
    mot = IntrinsicMotivation(K)                       # curiosidade sobre os objetos
    rng = np.random.default_rng(SEED)

    xs, perc, comm, surp = [], [], [], []
    for t in range(N_STEPS):
        # curiosidade escolhe o objeto (eps-greedy no learning progress)
        if rng.random() < 0.25:
            oi = int(rng.integers(K))
        else:
            oi = int(np.argmax(mot.learning_progress()))
        obj = P[oi] + NOISE * rng.standard_normal(D)

        eA = A.learn_perception(obj)                   # ambos aprendem a VER
        eB = B.learn_perception(obj)
        mot.update(oi, 0.5 * (eA + eB))                # curiosidade observa a surpresa

        spk, lis = (A, B) if t % 2 == 0 else (B, A)    # alternam quem fala
        c = spk.concept(obj)
        m = spk.speak(c)
        cr = lis.listen(m)
        lc = [lis.concept(P[j]) for j in range(K)]
        cands = [j for j in range(K) if lc[j] == cr]
        guess = int(rng.choice(cands)) if cands else int(rng.integers(K))
        spk.reinforce_speak(c, m, guess == oi)
        lis.learn_listen(m, lis.concept(obj))          # atenção conjunta

        if t % 100 == 0:
            xs.append(t)
            perc.append(100 * A.discriminability(P))
            comm.append(100 * 0.5 * (comm_acc(A, B, P) + comm_acc(B, A, P)))
            surp.append(np.mean([A.pc.prediction_error(P[j]) for j in range(K)]))
    xs = np.array(xs)
    # domínio perceptual: quanto da surpresa inicial já foi domada (medida honesta
    # do "aprender a ver"; a discriminabilidade já nasce alta com códigos aleatórios).
    pmast = 100.0 * (1.0 - np.array(surp) / surp[0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) co-emergência: percepção sobe primeiro, linguagem vem atrás.
    ax = axes[0, 0]
    ax.plot(xs, pmast, lw=2.2, color="#2ca02c", label="percepção (domínio do mundo: surpresa domada)")
    ax.plot(xs, comm, lw=2.2, color="#1f77b4", label="comunicação")
    ax.axvline(1200, color="#888", ls=":", lw=1.2)
    ax.text(1280, 8, "percepção assenta", color="#666", fontsize=8, rotation=90, va="bottom")
    ax.set_title("(a) Percepção e linguagem CO-EMERGEM (no mesmo laço, vivendo)")
    ax.set_xlabel("passos de vida"); ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8.5)

    # (b) aprender a VER: a surpresa cai.
    ax = axes[0, 1]
    ax.plot(xs, surp, lw=2, color="#9467bd")
    ax.set_title("(b) Aprendendo a ver: a surpresa cai")
    ax.set_xlabel("passos de vida"); ax.set_ylabel("erro de previsão (surpresa)")

    # (c) conceitos que o agente A formou enquanto vivia.
    ax = axes[1, 0]
    ax.imshow(_tile(A.pc.W, SIDE, N_LATENT), cmap="magma")
    ax.set_title("(c) Conceitos que A formou vivendo (campos receptivos)")
    ax.set_xticks([]); ax.set_yticks([])

    # (d) resumo.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M20 — o organismo vivo (a costura da cognição)\n\n"
            "Num laço só, dois agentes ao mesmo tempo:\n"
            "  • PERCEBEM e formam conceitos (M4/M9)\n"
            "  • LEMBRAM por replay (M8)\n"
            "  • são CURIOSOS (learning progress, M5)\n"
            "  • aprendem a FALAR sobre o que percebem (M16/M19)\n\n"
            f"• percepção final: {perc[-1]:.0f}% dos objetos distinguidos\n"
            f"• comunicação final: {comm[-1]:.0f}%\n\n"
            "Achado: a linguagem fica ATRÁS da percepção e só cola\n"
            "quando os conceitos estabilizam. Primeiro aprender a ver,\n"
            "depois aprender a falar — as duas nascendo da mesma vida.\n\n"
            "Honesto: escala de brinquedo; conceitos em formação são\n"
            "alvo móvel, então a linguagem custa mais a assentar.",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M20 — Um organismo que percebe, lembra, é curioso e fala, vivendo",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m20_living_mind.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"Percepção: {perc[0]:.0f}% -> {perc[-1]:.0f}%   Comunicação: {comm[0]:.0f}% -> {comm[-1]:.0f}%")
    print(f"Surpresa: {surp[0]:.3f} -> {surp[-1]:.3f}")


def _tile(W, side, cols):
    n = W.shape[1]; rows = int(np.ceil(n / cols))
    canvas = np.full((rows * (side + 1) - 1, cols * (side + 1) - 1), np.nan)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[r * (side + 1):r * (side + 1) + side, c * (side + 1):c * (side + 1) + side] = W[:, i].reshape(side, side)
    return canvas


if __name__ == "__main__":
    main()
