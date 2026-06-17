"""M15 — Gramática: aprender REGRAS sequenciais (Fase 3, rumo à sintaxe).

No M14 o brAIn decorou uma sequência fixa. Aqui ele aprende uma GRAMÁTICA — uma
máquina de estados com transições válidas e probabilidades — e GERA sequências
NOVAS que respeitam a estrutura. É o ensaio para a sintaxe da linguagem.

Lê do preditor temporal uma DISTRIBUIÇÃO sobre o próximo símbolo (softmax das
similaridades da previsão com cada protótipo), e compara com a gramática real.

Demonstra:
  (a) aprende a MATRIZ DE TRANSIÇÃO (qual símbolo pode seguir qual, com que prob);
  (b) GERA sequências gramaticais novas (quase só transições válidas);
  (c) bate o baseline aleatório em gramaticalidade.

Uso:   python experiments/m15_grammar.py
Salva: experiments/output/m15_grammar.png
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
D = 16
K = 6
TEMP = 0.12

# Gramática (1ª ordem): T[i,j] = P(próximo=j | atual=i). Estrutura: 0->{1,2};
# 1->3; 2->3; 3->{4,5}; 4->0; 5->0. (uma autômato com dois pontos de ramificação)
T = np.zeros((K, K))
T[0, 1] = 0.5; T[0, 2] = 0.5
T[1, 3] = 1.0
T[2, 3] = 1.0
T[3, 4] = 0.6; T[3, 5] = 0.4
T[4, 0] = 1.0
T[5, 0] = 1.0

SYMBOLS = np.array([0, 1, 2, 3, 4, 5])


def make_protos(seed=1):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (K, D))
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def sample_seq(n, rng, start=0):
    s = start; out = [s]
    for _ in range(n - 1):
        s = rng.choice(K, p=T[s])
        out.append(s)
    return out


def readout_dist(model, protos):
    """Distribuição prevista sobre o próximo símbolo, do contexto atual."""
    pred = model.predict()
    scores = protos @ pred
    z = np.exp((scores - scores.max()) / TEMP)
    return z / z.sum()


def learned_T(model, protos, rng):
    """Estima a matriz de transição aprendida: média da distribuição prevista por
    símbolo atual, ao longo de uma sequência de teste (sem aprender)."""
    seq = sample_seq(4000, rng)
    rows = [np.zeros(K) for _ in range(K)]
    cnt = np.zeros(K)
    model.reset_context()
    for t in range(len(seq) - 1):
        model.observe(protos[seq[t]], learn=False)   # atualiza contexto
        d = readout_dist(model, protos)
        rows[seq[t]] += d; cnt[seq[t]] += 1
    return np.array([rows[i] / max(cnt[i], 1) for i in range(K)])


def grammatical_frac(seq):
    pairs = [(seq[t], seq[t + 1]) for t in range(len(seq) - 1)]
    return np.mean([T[i, j] > 0 for i, j in pairs])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    protos = make_protos()
    rng = np.random.default_rng(SEED)

    # treina na gramática
    model = TemporalPredictiveCoder(D, decay=0.25, eta_w=0.05, seed=0)
    train = sample_seq(20000, rng)
    model.reset_context()
    for t in range(len(train) - 1):
        model.observe(protos[train[t]] + 0.02 * rng.standard_normal(D))

    learned = learned_T(model, protos, np.random.default_rng(1))

    # gera sequências novas amostrando da distribuição prevista
    gen_rng = np.random.default_rng(7)
    model.reset_context()
    model.observe(protos[0], learn=False)
    gen = [0]
    for _ in range(400):
        d = readout_dist(model, protos)
        nxt = int(gen_rng.choice(K, p=d))
        gen.append(nxt)
        model.observe(protos[nxt], learn=False)
    gen_gram = grammatical_frac(gen)

    rand_seq = list(np.random.default_rng(3).integers(K, size=400))
    rand_gram = grammatical_frac(rand_seq)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    im = ax.imshow(T, cmap="magma", vmin=0, vmax=1)
    ax.set_title("(a) Gramática REAL  T[atual, próximo]")
    ax.set_xlabel("próximo símbolo"); ax.set_ylabel("símbolo atual")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(learned, cmap="magma", vmin=0, vmax=1)
    ax.set_title("(b) Gramática APRENDIDA (distribuição prevista)")
    ax.set_xlabel("próximo símbolo"); ax.set_ylabel("símbolo atual")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    bars = ax.bar(["brAIn\n(gerado)", "aleatório"], [100 * gen_gram, 100 * rand_gram],
                  color=["#2ca02c", "#888888"])
    for b, v in zip(bars, [100 * gen_gram, 100 * rand_gram]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center",
                fontsize=11, fontweight="bold")
    ax.set_title("(c) Transições GRAMATICAIS nas sequências geradas")
    ax.set_ylabel("% de transições válidas"); ax.set_ylim(0, 105)

    ax = axes[1, 1]; ax.axis("off")
    # erro de reconstrução da matriz e exemplo de sequência gerada
    mae = np.mean(np.abs(T - learned))
    ax.text(0.0, 0.98,
            "M15 — gramática: aprende REGRAS, não uma sequência fixa\n\n"
            "Gramática (autômato):\n"
            "  0 → 1|2    1 → 3    2 → 3\n"
            "  3 → 4|5    4 → 0    5 → 0\n\n"
            f"• erro médio |real - aprendida|: {mae:.3f}\n"
            f"• gramaticalidade gerada: {100*gen_gram:.0f}%  (aleatório {100*rand_gram:.0f}%)\n\n"
            f"sequência gerada (nova, válida):\n  {gen[:24]}\n\n"
            "Aprendeu os PONTOS DE RAMIFICAÇÃO (0 e 3 viram dois\n"
            "caminhos) e gera frases novas que respeitam a regra.\n"
            "É o ensaio da sintaxe — a ponte para a linguagem.",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M15 — Gramática: aprende as regras da sequência e gera frases novas válidas",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m15_grammar.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"Erro médio matriz |real-aprendida|: {mae:.3f}")
    print(f"Gramaticalidade gerada: {100*gen_gram:.0f}%  (aleatório {100*rand_gram:.0f}%)")
    print(f"Exemplo gerado: {gen[:24]}")


if __name__ == "__main__":
    main()
