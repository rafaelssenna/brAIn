"""M33 — Frases relacionais e RECURSÃO (o poder infinito da linguagem).

Os marcos anteriores descreviam UM objeto. A linguagem é RECURSIVA: uma frase contém frases.
"A barra vermelha ACIMA do ponto azul" descreve DOIS objetos e uma RELAÇÃO — cada objeto é uma
sub-descrição. Aqui a cena tem dois objetos (forma × cor) e uma relação (acima/abaixo), e o
agente monta/entende a frase relacional inteira, composta de duas sub-frases + a relação.

São 162 cenas relacionais (9 objetos × 2 relações × 9 objetos). O agente generaliza para as
que nunca viu, porque entende as PARTES e a estrutura recursiva que as combina.

Pergunta científica (H16): o agente compõe e entende frases RECURSIVAS (objeto-relação-objeto)
e generaliza para cenas relacionais nunca vistas?

Uso:   python experiments/m33_relational_sentence.py
Salva: experiments/output/m33_relational_sentence.png

Honesto: recursão de UM nível, escala de brinquedo. É o MECANISMO da composição recursiva,
NÃO recursão arbitrariamente profunda nem sintaxe plena.

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
from brain import RelationalSentenceLearner, make_relational_language  # noqa: E402

np.seterr(all="ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
KS, KC, KR = 3, 3, 2
SCENES = [((s1, c1), r, (s2, c2)) for s1 in range(KS) for c1 in range(KC)
          for r in range(KR) for s2 in range(KS) for c2 in range(KC)]
N_STEPS = 6000
N_SEEDS = 6

SH = ["barra", "ponto", "anel"]
CO = ["vermelho", "verde", "azul"]
RE = ["acima", "abaixo"]
OBJ_W = KS + KC


def render(sent):
    out = []
    for w in sent:
        if w < KS:
            out.append(SH[w])
        elif w < KS + KC:
            out.append(CO[w - KS])
        else:
            out.append(RE[w - OBJ_W])
    return " ".join(out)


def teach(order, train, n=N_STEPS, seed=1):
    lang = make_relational_language(order, KS, KC)
    ag = RelationalSentenceLearner(KS, KC, KR, seed=seed)
    rng = np.random.default_rng(0)
    for _ in range(n):
        o1, r, o2 = train[int(rng.integers(len(train)))]
        ag.hear(lang(o1, r, o2), o1, r, o2)
    return ag, lang


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # generalização (média de sementes): treina em 70%, testa nas nunca vistas
    tr, te = [], []
    for sd in range(N_SEEDS):
        rng = np.random.default_rng(sd + 3)
        idx = rng.permutation(len(SCENES))
        ntr = int(0.7 * len(SCENES))
        train = [SCENES[i] for i in idx[:ntr]]; test = [SCENES[i] for i in idx[ntr:]]
        ag, lang = teach((0, 1), train, seed=sd + 1)
        tr.append(100 * np.mean([ag.say(*c) == lang(*c) for c in train]))
        te.append(100 * np.mean([ag.say(*c) == lang(*c) for c in test]))
    TR, TE = np.mean(tr), np.mean(te)

    # curva: generalização vs fração de cenas vistas
    fracs = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
    curve = []
    for f in fracs:
        accs = []
        for sd in range(N_SEEDS):
            rng = np.random.default_rng(sd + 30)
            idx = rng.permutation(len(SCENES))
            ntr = int(f * len(SCENES))
            train = [SCENES[i] for i in idx[:ntr]]; test = [SCENES[i] for i in idx[ntr:]]
            ag, lang = teach((0, 1), train, seed=sd + 1)
            accs.append(100 * np.mean([ag.say(*c) == lang(*c) for c in test]))
        curve.append(np.mean(accs))

    ag_ex, lang_ex = teach((0, 1), SCENES, seed=1)

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) generalização para cenas relacionais nunca vistas.
    ax = axes[0, 0]
    bars = ax.bar(["cenas de treino\n(70%)", "cenas relacionais\nNUNCA vistas (30%)"],
                  [TR, TE], color=["#1f77b4", "#2ca02c"])
    ax.set_ylabel("frase relacional certa %"); ax.set_ylim(0, 109)
    ax.set_title("(a) Compõe frases recursivas e generaliza")
    for b, v in zip(bars, [TR, TE]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=9)

    # (b) curva de generalização.
    ax = axes[0, 1]
    ax.plot([f * 100 for f in fracs], curve, lw=2.2, marker="o", color="#8a3ffc")
    ax.set_title("(b) Quanto basta ouvir para compor as relacionais nunca vistas")
    ax.set_xlabel("% das 162 cenas relacionais vistas no treino")
    ax.set_ylabel("frase certa nas nunca vistas %"); ax.set_ylim(0, 105)

    # (c) exemplos concretos (estrutura recursiva).
    ax = axes[1, 0]; ax.axis("off")
    lines = ["(c) Descrições RECURSIVAS (objeto - relação - objeto):", ""]
    for c in [((0, 0), 0, (1, 2)), ((2, 1), 1, (0, 0)), ((1, 2), 0, (2, 1))]:
        o1, r, o2 = c
        lines.append(f"cena: [{SH[o1[0]]} {CO[o1[1]]}] {RE[r]} [{SH[o2[0]]} {CO[o2[1]]}]")
        lines.append(f'   -> "{render(ag_ex.say(*c))}"')
        lines.append("")
    ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=8.8, family="monospace")

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M33 — frases relacionais e RECURSÃO\n\n"
            "A linguagem é recursiva: uma frase contém frases. Aqui a\n"
            "cena tem 2 objetos (forma × cor) e uma RELAÇÃO (acima/\n"
            "abaixo). A frase é sub(obj1) + relação + sub(obj2) — cada\n"
            "objeto é uma sub-descrição (o mesmo aprendiz, reusado).\n\n"
            f"• 162 cenas relacionais possíveis\n"
            f"• treino (70%):        {TR:.0f}%\n"
            f"• NUNCA vistas (30%):  {TE:.0f}%\n\n"
            "Achado (H16): o agente compõe e entende frases RECURSIVAS\n"
            "(estruturas dentro de estruturas) e generaliza para cenas\n"
            "relacionais nunca vistas — porque entende as PARTES (os\n"
            "objetos, a relação) e a estrutura que as combina. É o que\n"
            "dá à linguagem seu poder infinito: poucas peças, infinitas\n"
            "frases. Tudo ancorado, aprendido só ouvindo.\n\n"
            "Honesto: recursão de UM nível, escala de brinquedo;\n"
            "mecanismo da composição recursiva, não recursão profunda\n"
            "arbitrária nem sintaxe plena.",
            transform=ax.transAxes, va="top", fontsize=8.2, family="monospace")

    fig.suptitle("brAIn · M33 — Recursão: o agente descreve estruturas dentro de estruturas ('a barra acima do ponto')",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m33_relational_sentence.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Treino {TR:.0f}% | NUNCA vistas {TE:.0f}%")
    print(f"Exemplo: \"{render(ag_ex.say((0,0),0,(1,2)))}\"")


if __name__ == "__main__":
    main()
