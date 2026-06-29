"""M31 — Correferência: resolver 'dela' entre VÁRIOS objetos (a ambiguidade do diálogo).

O M30 manteve UM objeto em foco, então "dela" era trivial. Mas uma conversa real fala de
VÁRIOS objetos, e "dela" fica AMBÍGUO: qual delas? O cérebro resolve com pistas — a mais
recente/saliente, ou a que casa com uma descrição. Aqui a cena tem vários objetos e o agente
precisa escolher o referente CERTO.

    cena: [barraH no topo] e [barraV na base]
    P: "e a posição DELA?"        -> resolve a barraV (a recém-mencionada), diz "base"
    P: "a barraH, a posição dela?" -> resolve pela descrição (a barraH), diz "topo"

Pergunta científica (H14): com vários candidatos, o agente resolve 'dela' escolhendo o
referente certo (pela recência ou pela descrição) — algo que escolher ao acaso não consegue?

Provas (método do projeto):
  • RECÊNCIA resolve o referente certo (o mais recente);
  • BASELINE aleatório (escolhe um candidato ao acaso) cai a ~acaso entre eles;
  • DESCRIÇÃO desambigua: escolhe pelo atributo dito, mesmo não sendo o mais recente.

Uso:   python experiments/m31_coreference.py
Salva: experiments/output/m31_coreference.png

Honesto: escala de brinquedo (2 objetos, recência simples, descrição por 1 atributo). É o
MECANISMO da correferência, NÃO resolução plena (sem gênero/número, sem cadeias longas, sem
inferência pragmática) — horizonte distante, provável híbrido.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import CoreferenceDialogue  # noqa: E402

np.seterr(all="ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
K_SHAPE = 3
K_POS = 3
ALL = [(s, p) for s in range(K_SHAPE) for p in range(K_POS)]
N_GROUND = 8000
N_SEEDS = 6

SHAPE_WORDS = ["barraH", "barraV", "diag"]
POS_WORDS = ["topo", "meio", "base"]


def trained(strategy, seed):
    g = CoreferenceDialogue(K_SHAPE, K_POS, strategy=strategy, seed=seed)
    g.train_grounding(ALL, N_GROUND, rng_seed=seed + 50)
    return g


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pairs = [(o1, o2) for o1 in ALL for o2 in ALL if o1 != o2][:40]
    # descrições por forma (objetos com formas diferentes para a descrição desambiguar)
    triples = []
    for o1 in ALL:
        for o2 in ALL:
            if o1[0] != o2[0]:
                triples.append((o1, o2, ("shape", o1[0])))    # descreve o1 (o mais ANTIGO)
    triples = triples[:40]

    rec_pos, rnd_pos, desc = [], [], []
    for sd in range(N_SEEDS):
        rec_pos.append(100 * trained("recency", sd + 1).eval_recency(pairs, ask="pos"))
        rnd_pos.append(100 * trained("random", sd + 1).eval_recency(pairs, ask="pos"))
        desc.append(100 * trained("recency", sd + 1).eval_described(triples, ask="pos"))
    rec, rnd, dsc = np.mean(rec_pos), np.mean(rnd_pos), np.mean(desc)
    chance = 50.0          # 2 candidatos

    # exemplos concretos
    g = trained("recency", 1)
    examples = []
    for (o1, o2) in [((0, 0), (1, 2)), ((2, 1), (0, 0))]:
        g.reset(); g.mention(o1); g.mention(o2)
        ref_r, val_r = g.answer_ref(ask="pos")
        ref_d, val_d = g.answer_ref(ask="pos", describe=("shape", o1[0]))
        examples.append((o1, o2, ref_r, val_r, ref_d, val_d))

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) recência resolve; aleatório cai ao acaso.
    ax = axes[0, 0]
    bars = ax.bar(["RECÊNCIA\n(o mais recente)", "ALEATÓRIO\n(baseline)"], [rec, rnd],
                  color=["#1f77b4", "#d62728"])
    ax.axhline(chance, color="#999", ls=":", lw=1, label=f"acaso (2 candidatos, {chance:.0f}%)")
    ax.set_ylabel("referência resolvida %"); ax.set_ylim(0, 109)
    ax.set_title("(a) Resolver 'dela' entre 2 objetos: recência vs acaso")
    ax.legend(loc="lower right", fontsize=8.5)
    for b, v in zip(bars, [rec, rnd]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=9)

    # (b) a descrição desambigua (escolhe pelo atributo, não pela recência).
    ax = axes[0, 1]
    bars = ax.bar(['"DELA" (recência)', '"a barraX, dela"\n(descrição)'], [rec, dsc],
                  color=["#1f77b4", "#2ca02c"])
    ax.set_ylabel("referência resolvida %"); ax.set_ylim(0, 109)
    ax.set_title("(b) A descrição escolhe pelo atributo (mesmo o mais antigo)")
    for b, v in zip(bars, [rec, dsc]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=9)

    # (c) exemplos concretos (texto).
    ax = axes[1, 0]; ax.axis("off")
    lines = ["(c) Resolvendo a ambiguidade (2 objetos na cena):", ""]
    for (o1, o2, refr, valr, refd, vald) in examples:
        lines.append(f"cena: [{SHAPE_WORDS[o1[0]]} {POS_WORDS[o1[1]]}] e "
                     f"[{SHAPE_WORDS[o2[0]]} {POS_WORDS[o2[1]]}]")
        lines.append(f'   "e a posição DELA?"      -> {SHAPE_WORDS[refr[0]]}: "{POS_WORDS[valr]}"')
        lines.append(f'   "a {SHAPE_WORDS[o1[0]]}, posição dela?" -> {SHAPE_WORDS[refd[0]]}: "{POS_WORDS[vald]}"')
        lines.append("")
    ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=8.8, family="monospace")

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M31 — correferência: resolver 'dela' entre vários\n\n"
            "A cena tem VÁRIOS objetos, então 'dela' é AMBÍGUO: qual\n"
            "delas? O agente mantém uma pilha de foco e resolve a\n"
            "referência — pela RECÊNCIA (a recém-mencionada) ou por\n"
            "DESCRIÇÃO ('a barraX...'), que escolhe pelo atributo.\n\n"
            f"• recência:   {rec:.0f}% (resolve o referente certo)\n"
            f"• aleatório:  {rnd:.0f}% (acaso entre 2 candidatos ~{chance:.0f}%)\n"
            f"• descrição:  {dsc:.0f}% (desambigua pelo atributo dito)\n\n"
            "Achado (H14): com vários candidatos, o agente resolve\n"
            "'dela' escolhendo o referente CERTO — pela recência ou\n"
            "pela descrição. Escolher ao acaso cai à metade. É o\n"
            "começo de entender a AMBIGUIDADE da linguagem: a quem\n"
            "um pronome se refere depende do contexto e das pistas.\n\n"
            "Honesto: 2 objetos, recência/descrição simples, escala de\n"
            "brinquedo; mecanismo da correferência, não resolução plena\n"
            "(sem gênero/número, sem cadeias longas, sem pragmática).",
            transform=ax.transAxes, va="top", fontsize=8.2, family="monospace")

    fig.suptitle("brAIn · M31 — Correferência: o agente resolve a quem 'dela' se refere entre vários objetos",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m31_coreference.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Recencia:  {rec:.0f}% | Aleatorio: {rnd:.0f}% (acaso ~{chance:.0f}%) | Descricao: {dsc:.0f}%")


if __name__ == "__main__":
    main()
