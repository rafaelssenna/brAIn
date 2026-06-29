"""M35 — Descobrir a gramática por ESTATÍSTICA pura (aquisição não-supervisionada).

Até aqui um "professor" dava atenção conjunta. Aqui é como um bebê ouvindo a língua ao redor:
o agente recebe SÓ um corpus de FRASES (sequências de palavras), sem NENHUM rótulo, e descobre
sozinho (1) as CLASSES de palavras (cores juntas, formas juntas, posições juntas) e (2) a
GRAMÁTICA (quais sequências de classes são válidas) — pela ESTATÍSTICA das co-ocorrências, como
na aprendizagem estatística de linguagem em bebês (Saffran, Aslin & Newport, 1996).

Pergunta científica (H18): o agente descobre as classes de palavras e a gramática de um corpus
SÓ pela estatística — sem rótulos — e distingue frases gramaticais de embaralhadas?

Uso:   python experiments/m35_grammar_discovery.py
Salva: experiments/output/m35_grammar_discovery.png

Honesto: vocabulário pequeno, gramática regular (sem recursão), escala de brinquedo. É o
MECANISMO da descoberta estatística de classes/gramática, NÃO indução de gramática plena.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import GrammarDiscovery  # noqa: E402

np.seterr(all="ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SHAPES, COLORS, POS = [0, 1, 2], [3, 4, 5], [6, 7, 8]
TRUE = {**{w: 0 for w in SHAPES}, **{w: 1 for w in COLORS}, **{w: 2 for w in POS}}
N_SEEDS = 6
LABELS_TXT = ["barra", "ponto", "anel", "vermelho", "verde", "azul", "topo", "meio", "base"]


def run(seed, n_corpus=3000):
    rng = np.random.default_rng(seed)
    corpus = [[rng.choice(SHAPES), rng.choice(COLORS), rng.choice(POS)] for _ in range(n_corpus)]
    gd = GrammarDiscovery(9, 3, seed=seed)
    gd.observe_corpus(corpus)
    gd.discover_grammar()
    perms = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]
    good = np.mean([gd.is_grammatical([rng.choice(SHAPES), rng.choice(COLORS), rng.choice(POS)])
                    for _ in range(100)])
    def bad():
        base = [rng.choice(SHAPES), rng.choice(COLORS), rng.choice(POS)]
        p = perms[rng.integers(len(perms))]
        return [base[i] for i in p]
    badrej = np.mean([not gd.is_grammatical(bad()) for _ in range(100)])
    return gd, 100 * gd.class_purity(TRUE), 100 * good, 100 * badrej


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    res = np.array([run(s + 1)[1:] for s in range(N_SEEDS)])
    purity, good, badrej = res.mean(axis=0)

    # curva: pureza vs tamanho do corpus
    sizes = [100, 300, 600, 1000, 2000, 3000]
    pur_curve = [np.mean([run(s + 1, n)[1] for s in range(N_SEEDS)]) for n in sizes]

    gd_ex = run(1)[0]

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) métricas da descoberta não-supervisionada.
    ax = axes[0, 0]
    bars = ax.bar(["classes\ndescobertas", "reconhece\ncorretas", "rejeita\nembaralhadas"],
                  [purity, good, badrej], color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    ax.set_ylabel("%"); ax.set_ylim(0, 109)
    ax.set_title("(a) Descobre classes e gramática SÓ ouvindo (sem rótulos)")
    for b, v in zip(bars, [purity, good, badrej]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=9)

    # (b) as classes descobertas (matriz palavra -> classe).
    ax = axes[0, 1]
    M = np.zeros((9, 3))
    for w in range(9):
        M[w, gd_ex.labels[w]] = 1
    ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_yticks(range(9)); ax.set_yticklabels(LABELS_TXT, fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(["classe A", "classe B", "classe C"], fontsize=8)
    ax.set_title("(b) Classes descobertas: formas/cores/posições separam sozinhas")

    # (c) a gramática aprendida (transições entre classes).
    ax = axes[1, 0]
    im = ax.imshow(gd_ex.trans, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("classe seguinte"); ax.set_ylabel("classe atual")
    ax.set_title("(c) Gramática aprendida: transições válidas entre classes")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{gd_ex.trans[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if gd_ex.trans[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M35 — descobrir a gramática por estatística pura\n\n"
            "Como um bebê ouvindo a língua: o agente recebe SÓ frases\n"
            "(sem nenhum rótulo) e descobre, pela ESTATÍSTICA das\n"
            "co-ocorrências, as CLASSES de palavras (palavras com\n"
            "contextos parecidos) e a GRAMÁTICA (transições válidas).\n\n"
            f"• classes descobertas (pureza):  {purity:.0f}%\n"
            f"• reconhece frases corretas:     {good:.0f}%\n"
            f"• rejeita frases embaralhadas:   {badrej:.0f}%\n\n"
            "Achado (H18): sem rótulos, sem professor, o agente recupera\n"
            "as classes (formas/cores/posições separam sozinhas) e a\n"
            "ordem da língua só pela estatística — como na aprendizagem\n"
            "estatística de linguagem em bebês (Saffran et al. 1996). E\n"
            "usa isso para distinguir o gramatical do embaralhado.\n\n"
            "Honesto: vocabulário pequeno, gramática regular (sem\n"
            "recursão), escala de brinquedo; mecanismo da descoberta\n"
            "estatística de classes/gramática, não indução plena.",
            transform=ax.transAxes, va="top", fontsize=8.2, family="monospace")

    fig.suptitle("brAIn · M35 — Descoberta não-supervisionada: o agente acha as classes de palavras e a gramática só ouvindo",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m35_grammar_discovery.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Pureza {purity:.0f}% | reconhece corretas {good:.0f}% | rejeita embaralhadas {badrej:.0f}%")


if __name__ == "__main__":
    main()
