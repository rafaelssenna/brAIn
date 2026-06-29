"""M30 — Diálogo de VÁRIOS TURNOS com CONTEXTO (a conversa coerente no tempo).

O M29 fez um turno isolado. Uma conversa de verdade tem MEMÓRIA: o que foi dito antes
importa, e a gente se refere a isso. Aqui o diálogo tem vários turnos sobre o mesmo objeto:

    P: "que forma?"          R: "barraH"     (o objeto entra em FOCO)
    P: "e a posição dela?"   R: "topo"       (referência ao foco — nada é remostrado)

A capacidade NOVA: manter um OBJETO EM FOCO na memória e RESOLVER A REFERÊNCIA ("dela") ao
que ficou em foco no turno anterior. É o que liga linguagem (M29) à memória (M8) e torna a
conversa COERENTE no tempo.

Pergunta científica (H13): um agente sustenta um diálogo de vários turnos, resolvendo uma
referência ao objeto em foco — algo que um agente SEM memória de contexto não consegue?

Provas (método do projeto):
  • COM contexto resolve "e a posição/forma dela?";
  • BASELINE sem contexto (cada turno isolado) cai ao ACASO na referência — prova que a
    memória de foco é o que sustenta a conversa coerente;
  • o grounding de forma/posição é o do M29 (ancorado).

Uso:   python experiments/m30_contextual_dialogue.py
Salva: experiments/output/m30_contextual_dialogue.png

Honesto: escala de brinquedo (1 objeto em foco por vez, referência simples). É o MECANISMO
do contexto de diálogo, NÃO conversa plena (sem múltiplos referentes, sem correferência
ambígua, sem pragmática rica) — horizonte distante, provável híbrido.

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
from brain import (ContextualDialogue, ASK_SHAPE_NEW, ASK_POS_REF, ASK_SHAPE_REF)  # noqa: E402

np.seterr(all="ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
K_SHAPE = 3
K_POS = 3
ALL = [(s, p) for s in range(K_SHAPE) for p in range(K_POS)]
N_GROUND = 8000
N_SEEDS = 6

SHAPE_WORDS = ["barraH", "barraV", "diag"]
POS_WORDS = ["topo", "meio", "base"]


def trained(use_ctx, seed):
    g = ContextualDialogue(K_SHAPE, K_POS, use_context=use_ctx, seed=seed)
    g.train_grounding(ALL, N_GROUND, rng_seed=seed + 50)
    return g


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- com vs sem contexto: resolução de referência (média de sementes) ---
    ctx_pos, ctx_sh, no_pos, no_sh = [], [], [], []
    for sd in range(N_SEEDS):
        gc = trained(True, sd + 1)
        gn = trained(False, sd + 1)
        ctx_pos.append(100 * gc.eval_reference(ALL, ask_pos_ref=True))
        ctx_sh.append(100 * gc.eval_reference(ALL, ask_pos_ref=False))
        no_pos.append(100 * gn.eval_reference(ALL, ask_pos_ref=True))
        no_sh.append(100 * gn.eval_reference(ALL, ask_pos_ref=False))
    cP, cS = np.mean(ctx_pos), np.mean(ctx_sh)
    nP, nS = np.mean(no_pos), np.mean(no_sh)
    chanceP, chanceS = 100.0 / K_POS, 100.0 / K_SHAPE

    # diálogo de exemplo (agente com contexto)
    g = trained(True, 1)
    examples = []
    for (s, p) in [(0, 0), (1, 2), (2, 1)]:
        g.reset_focus()
        w1, _ = g.step(ASK_SHAPE_NEW, scene=(s, p), learn=False, explore=False)
        a1 = SHAPE_WORDS[g.dlg.interpret(w1, 0)]
        w2, r2 = g.step(ASK_POS_REF, true_referent=(s, p), learn=False, explore=False)
        a2 = POS_WORDS[g.dlg.interpret(w2, 1)]
        examples.append((SHAPE_WORDS[s], POS_WORDS[p], a1, a2, r2))

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) resolução de referência: com vs sem contexto.
    ax = axes[0, 0]
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, [cP, cS], w, color="#1f77b4", label="COM memória de contexto")
    ax.bar(x + w / 2, [nP, nS], w, color="#d62728", label="SEM contexto (turnos isolados)")
    ax.axhline(chanceP, color="#999", ls=":", lw=1, label=f"acaso (~{chanceP:.0f}%)")
    ax.set_xticks(x); ax.set_xticklabels(['"e a posição dela?"', '"e a forma dela?"'])
    ax.set_ylabel("referência resolvida %"); ax.set_ylim(0, 109)
    ax.set_title("(a) Só com MEMÓRIA de contexto o agente resolve 'dela'")
    ax.legend(loc="lower center", fontsize=8)
    for xi, v in zip([x[0] - w / 2, x[1] - w / 2], [cP, cS]):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    for xi, v in zip([x[0] + w / 2, x[1] + w / 2], [nP, nS]):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)

    # (b) o ganho do contexto, por referência.
    ax = axes[0, 1]
    gains = [cP - nP, cS - nS]
    ax.bar(['"posição dela"', '"forma dela"'], gains, color="#2ca02c")
    ax.set_ylabel("ganho do contexto (pontos %)")
    ax.set_title("(b) O quanto a memória de contexto adiciona")
    for i, v in enumerate(gains):
        ax.text(i, v + 0.5, f"+{v:.0f}", ha="center", fontsize=9)

    # (c) diálogo de exemplo (texto).
    ax = axes[1, 0]; ax.axis("off")
    lines = ["(c) Um diálogo de 2 turnos (agente com contexto):", ""]
    for (sh, po, a1, a2, ok) in examples:
        lines.append(f"cena: [{sh} no {po}]")
        lines.append(f'   P: que forma?          R: "{a1}"')
        lines.append(f'   P: e a posição DELA?   R: "{a2}"  {"OK" if ok else "x"}')
        lines.append("")
    ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=9.3, family="monospace")

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M30 — diálogo de vários turnos com CONTEXTO\n\n"
            "Uma conversa tem memória: o objeto sobre o qual se falou\n"
            "fica EM FOCO, e turnos seguintes se referem a ele ('dela')\n"
            "sem remostrar nada. O agente lembra o foco e RESOLVE A\n"
            "REFERÊNCIA. Liga linguagem (M29) à memória (M8).\n\n"
            f"• COM contexto:  posição dela {cP:.0f}%  |  forma dela {cS:.0f}%\n"
            f"• SEM contexto:  posição dela {nP:.0f}%  |  forma dela {nS:.0f}%\n"
            f"                 (acaso ~{chanceP:.0f}%)\n\n"
            "Achado (H13): a MEMÓRIA de foco é o que sustenta a conversa\n"
            "coerente no tempo. Sem ela, cada turno é isolado e o agente\n"
            "não sabe a quem 'dela' se refere — cai ao acaso. Com ela,\n"
            "resolve a referência e mantém o assunto. É o começo da\n"
            "COERÊNCIA do diálogo: lembrar do que se falou.\n\n"
            "Honesto: 1 objeto em foco por vez, referência simples,\n"
            "escala de brinquedo; mecanismo do contexto de diálogo, não\n"
            "conversa plena (sem múltiplos referentes nem correferência\n"
            "ambígua).",
            transform=ax.transAxes, va="top", fontsize=8.2, family="monospace")

    fig.suptitle("brAIn · M30 — Diálogo de vários turnos: o agente lembra o objeto em foco e resolve a referência ('dela')",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m30_contextual_dialogue.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"COM contexto:  posicao dela {cP:.0f}% | forma dela {cS:.0f}%")
    print(f"SEM contexto:  posicao dela {nP:.0f}% | forma dela {nS:.0f}% (acaso ~{chanceP:.0f}%)")
    print(f"Ganho do contexto: +{cP-nP:.0f} (posicao), +{cS-nS:.0f} (forma)")


if __name__ == "__main__":
    main()
