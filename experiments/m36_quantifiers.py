"""M36 — Quantificadores: "todos", "algum", "nenhum" (semântica de verdade).

A linguagem faz AFIRMAÇÕES sobre CONJUNTOS, com QUANTIFICADORES: "TODOS são vermelhos",
"ALGUM é vermelho", "NENHUM é vermelho". Avaliar isso não é descrever — é checar uma frase
CONTRA o mundo (semântica de valor de verdade): percorrer o conjunto e aplicar ∀ / ∃ / ¬∃.

O agente percebe a cor de cada objeto da cena (seu grounding) e responde VERDADEIRO/FALSO
para a afirmação (quantificador, cor).

Pergunta científica (H19): o agente avalia frases QUANTIFICADAS corretamente — aplicando a
lógica de ∀/∃/¬∃ sobre o conjunto percebido — em vez de só nomear?

Uso:   python experiments/m36_quantifiers.py
Salva: experiments/output/m36_quantifiers.png

Honesto: 3 quantificadores, conjuntos pequenos, 1 propriedade, escala de brinquedo. É o
MECANISMO da quantificação, NÃO semântica plena (sem aninhamento nem escopo).

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
from brain import QuantifierGame, ALL, SOME, NONE  # noqa: E402

np.seterr(all="ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
KC = 3
N_GROUND = 5000
N_SEEDS = 6
CO = ["vermelho", "verde", "azul"]
QN = ["todos", "algum", "nenhum"]


def trained(logic, seed):
    g = QuantifierGame(KC, correct_logic=logic, seed=seed)
    g.train_grounding(N_GROUND, rng_seed=seed + 50)
    return g


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    def agg(logic):
        per = {q: [] for q in QN}
        for sd in range(N_SEEDS):
            a = trained(logic, sd + 1).eval_accuracy(rng_seed=sd)
            for q in QN:
                per[q].append(a[q])
        return {q: 100 * np.mean(v) for q, v in per.items()}

    good = agg(True)
    bad = agg(False)

    g = trained(True, 1)
    examples = []
    rng = np.random.default_rng(0)
    for q, color, scene in [(ALL, 0, [0, 0, 0]), (SOME, 0, [1, 0, 2]), (NONE, 0, [1, 2, 1]),
                            (ALL, 0, [0, 1, 0])]:
        pred = g.evaluate(q, color, scene)
        examples.append((q, color, scene, pred))

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) acurácia por quantificador: lógica correta vs confusa.
    ax = axes[0, 0]
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, [good[q] for q in QN], w, color="#1f77b4", label="lógica correta")
    ax.bar(x + w / 2, [bad[q] for q in QN], w, color="#d62728", label="confunde os quantificadores")
    ax.set_xticks(x); ax.set_xticklabels([f'"{q}"' for q in QN])
    ax.set_ylabel("acerto V/F %"); ax.set_ylim(0, 109)
    ax.set_title("(a) Avalia ∀ / ∃ / ¬∃ corretamente sobre o conjunto")
    ax.legend(loc="lower center", fontsize=8)
    for xi, q in zip(x - w / 2, QN):
        ax.text(xi, good[q] + 1, f"{good[q]:.0f}", ha="center", fontsize=8)
    for xi, q in zip(x + w / 2, QN):
        ax.text(xi, bad[q] + 1, f"{bad[q]:.0f}", ha="center", fontsize=8)

    # (b) a tabela-verdade dos quantificadores.
    ax = axes[0, 1]; ax.axis("off")
    ax.text(0.5, 0.5,
            "Os quantificadores são LÓGICA:\n\n"
            "  todos(c)  = TODO objeto tem cor c   (∀)\n"
            "  algum(c)  = ALGUM objeto tem cor c  (∃)\n"
            "  nenhum(c) = NENHUM tem cor c        (¬∃)\n\n"
            "Avaliar = checar a frase CONTRA a cena\n"
            "(semântica de valor de verdade), não\n"
            "descrever. O agente percebe as cores e\n"
            "aplica a lógica sobre o conjunto.",
            transform=ax.transAxes, va="center", ha="center", fontsize=9.5, family="monospace")
    ax.set_title("(b) Quantificador = operação lógica sobre o conjunto")

    # (c) exemplos concretos.
    ax = axes[1, 0]; ax.axis("off")
    lines = ["(c) Avaliando afirmações contra a cena:", ""]
    for (q, color, scene, pred) in examples:
        cena = "[" + ", ".join(CO[c] for c in scene) + "]"
        lines.append(f"cena: {cena}")
        lines.append(f'   "{QN[q]} {CO[color]}?"  ->  {"VERDADEIRO" if pred else "FALSO"}')
        lines.append("")
    ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=9.0, family="monospace")

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    overall_g = np.mean(list(good.values()))
    overall_b = np.mean(list(bad.values()))
    ax.text(0.0, 0.98,
            "M36 — quantificadores: todos / algum / nenhum\n\n"
            "A linguagem afirma sobre CONJUNTOS com quantificadores.\n"
            "Avaliar 'todos são vermelhos?' é checar a frase CONTRA o\n"
            "mundo (semântica de verdade) — percorrer o conjunto e\n"
            "aplicar ∀ / ∃ / ¬∃, não só nomear.\n\n"
            f"• lógica correta:  todos {good['todos']:.0f}% | algum {good['algum']:.0f}% | "
            f"nenhum {good['nenhum']:.0f}%  (geral {overall_g:.0f}%)\n"
            f"• confusa:         todos {bad['todos']:.0f}% | algum {bad['algum']:.0f}% | "
            f"nenhum {bad['nenhum']:.0f}%  (geral {overall_b:.0f}%)\n\n"
            "Achado (H19): o agente avalia frases QUANTIFICADAS — aplica\n"
            "a lógica de ∀/∃/¬∃ sobre o conjunto que percebe. Quem\n"
            "confunde os quantificadores (trata tudo como 'algum') erra\n"
            "'todos' e 'nenhum' sistematicamente. É semântica de verdade:\n"
            "a frase é VERDADEIRA ou FALSA conforme o mundo — raciocinar,\n"
            "não descrever.\n\n"
            "Honesto: 3 quantificadores, conjuntos pequenos, 1 propriedade,\n"
            "escala de brinquedo; mecanismo da quantificação, não semântica\n"
            "plena (sem aninhamento nem escopo).",
            transform=ax.transAxes, va="top", fontsize=8.0, family="monospace")

    fig.suptitle("brAIn · M36 — Quantificadores: o agente avalia 'todos/algum/nenhum' contra a cena (semântica de verdade)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m36_quantifiers.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Logica correta:  todos {good['todos']:.0f}% algum {good['algum']:.0f}% nenhum {good['nenhum']:.0f}% (geral {overall_g:.0f}%)")
    print(f"Confusa:         todos {bad['todos']:.0f}% algum {bad['algum']:.0f}% nenhum {bad['nenhum']:.0f}% (geral {overall_b:.0f}%)")


if __name__ == "__main__":
    main()
