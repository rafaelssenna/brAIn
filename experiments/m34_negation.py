"""M34 — Negação e contraste (operadores lógicos sobre a referência).

Até aqui o agente descrevia o que VÊ. A linguagem também NEGA: "NÃO o vermelho", "o OUTRO".
Isso é um OPERADOR LÓGICO que transforma o conjunto de referentes — "não X" seleciona o
complemento de X. Entender isso é raciocinar sobre a linguagem, não só nomear.

Cena com 2 objetos de cores distintas; o falante aponta um por NEGAÇÃO do outro ("não o
vermelho" -> o azul). O ouvinte decodifica a cor negada e escolhe o objeto que NÃO a tem.

Pergunta científica (H17): o agente entende a NEGAÇÃO — identificar o objeto pelo que ele NÃO
é (o complemento)? Um agente que ignora o "não" escolhe exatamente o errado.

Uso:   python experiments/m34_negation.py
Salva: experiments/output/m34_negation.png

Honesto: negação simples sobre 2 objetos e 1 atributo, escala de brinquedo. É o MECANISMO do
operador de negação, NÃO lógica plena (sem quantificadores, sem escopo aninhado).

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
from brain import NegationGame  # noqa: E402

np.seterr(all="ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
KC = 3
PAIRS = [(a, b) for a in range(KC) for b in range(KC) if a != b]
N_GROUND = 6000
N_SEEDS = 6
CO = ["vermelho", "verde", "azul"]


def trained(neg, seed):
    g = NegationGame(KC, understand_negation=neg, seed=seed)
    g.train_grounding(N_GROUND, rng_seed=seed + 50)
    return g


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    yes = [100 * trained(True, sd + 1).eval_negation(PAIRS) for sd in range(N_SEEDS)]
    no = [100 * trained(False, sd + 1).eval_negation(PAIRS) for sd in range(N_SEEDS)]
    Y, N = np.mean(yes), np.mean(no)
    chance = 50.0          # 2 objetos

    g = trained(True, 1)
    examples = []
    for (a, b) in [(0, 2), (1, 0), (2, 1)]:
        objs = [a, b]
        w = g.color_word(a)
        ch = g.resolve_negation(w, objs)
        examples.append((a, b, objs[ch]))

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) entender a negação vs ignorá-la.
    ax = axes[0, 0]
    bars = ax.bar(["ENTENDE a negação\n('não X' -> o outro)", "IGNORA a negação\n(baseline)"],
                  [Y, N], color=["#1f77b4", "#d62728"])
    ax.axhline(chance, color="#999", ls=":", lw=1, label=f"acaso ({chance:.0f}%)")
    ax.set_ylabel("aponta o objeto certo %"); ax.set_ylim(0, 109)
    ax.set_title("(a) 'Não o vermelho' -> o agente escolhe o COMPLEMENTO")
    ax.legend(loc="center right", fontsize=8.5)
    for b, v in zip(bars, [Y, N]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=9)

    # (b) o significado da negação: inverte a escolha.
    ax = axes[0, 1]; ax.axis("off")
    ax.text(0.5, 0.5,
            "A negação é um OPERADOR:\n\n"
            "  'não X'  =  o complemento de X\n\n"
            "Entender:  'nao vermelho' -> AZUL    (certo)\n"
            "Ignorar:   'nao vermelho' -> vermelho (errado)\n\n"
            "O agente que ENTENDE escolhe o objeto pelo\n"
            "que ele NÃO é. O que IGNORA pega exatamente\n"
            "o errado (o objeto negado) — por isso 0%.",
            transform=ax.transAxes, va="center", ha="center", fontsize=10, family="monospace")
    ax.set_title("(b) Negação inverte a referência")

    # (c) exemplos concretos.
    ax = axes[1, 0]; ax.axis("off")
    lines = ["(c) Apontar por negação (2 objetos na cena):", ""]
    for (a, b, chosen) in examples:
        lines.append(f"objetos: [{CO[a]}] e [{CO[b]}]")
        lines.append(f'   "não o {CO[a]}"  -> aponta o [{CO[chosen]}]')
        lines.append("")
    ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=9.3, family="monospace")

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M34 — negação e contraste\n\n"
            "A linguagem não só descreve: ela NEGA ('não o vermelho').\n"
            "Negar é um OPERADOR LÓGICO — 'não X' = o complemento de X.\n"
            "O ouvinte decodifica a cor negada e escolhe, entre os 2\n"
            "objetos, o que NÃO a tem.\n\n"
            f"• entende a negação:  {Y:.0f}%\n"
            f"• ignora a negação:   {N:.0f}% (pega o objeto NEGADO, o errado)\n"
            f"• acaso (2 objetos):  {chance:.0f}%\n\n"
            "Achado (H17): o agente entende a NEGAÇÃO — identificar um\n"
            "objeto pelo que ele NÃO é. Isso é raciocinar sobre a\n"
            "linguagem (um operador que transforma o conjunto de\n"
            "referentes), não só nomear o que se vê. Quem ignora o\n"
            "'não' escolhe exatamente o errado — o contraste é total.\n\n"
            "Honesto: negação simples sobre 2 objetos e 1 atributo,\n"
            "escala de brinquedo; mecanismo do operador de negação,\n"
            "não lógica plena (sem quantificadores nem escopo aninhado).",
            transform=ax.transAxes, va="top", fontsize=8.2, family="monospace")

    fig.suptitle("brAIn · M34 — Negação: o agente aponta um objeto pelo que ele NÃO é ('não o vermelho')",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m34_negation.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Entende negacao: {Y:.0f}% | Ignora: {N:.0f}% (acaso {chance:.0f}%)")
    print(f"Exemplo: objetos [{CO[0]},{CO[2]}], 'nao {CO[0]}' -> {CO[examples[0][2]]}")


if __name__ == "__main__":
    main()
