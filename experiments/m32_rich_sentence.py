"""M32 — Frases mais ricas: descrição de 3 atributos (a produtividade da linguagem).

O M27 montou frases de 2 atributos. Uma descrição de verdade é mais rica: "a barra VERMELHA
no TOPO" — forma, COR e posição. Aqui o agente percebe 3 atributos e monta/entende uma frase
de 3 palavras, descobrindo SÓ OUVINDO o grounding (palavra↔atributo) e a ORDEM das palavras.

Com 3 atributos × 3 valores há 27 cenas possíveis. Treinando numa fração e testando nas NUNCA
vistas, prova-se a PRODUTIVIDADE da linguagem: montar descrições novas, que nunca se ouviu.

Pergunta científica (H15): o agente monta descrições COMPOSICIONAIS de 3 atributos, generaliza
para cenas nunca vistas, e o mesmo mecanismo aprende ordens de idiomas diferentes?

Provas (método do projeto):
  • produz/entende a frase de 3 atributos (PT e outra ordem);
  • GENERALIZA para as cenas NUNCA vistas (a maior parte das 27);
  • BASELINE de ordem aleatória não aprende a sintaxe (controle);
  • curva de generalização vs nº de cenas vistas no treino (quanto basta para compor o resto).

Uso:   python experiments/m32_rich_sentence.py
Salva: experiments/output/m32_rich_sentence.png

Honesto: escala de brinquedo (3 atributos, 3 valores, frase de tamanho fixo). É o MECANISMO
da descrição composicional rica, NÃO fluência, NÃO recursão, NÃO tamanho variável.

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
from brain import RichSentenceLearner, make_rich_language  # noqa: E402

np.seterr(all="ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
K = [3, 3, 3]                 # forma, cor, posição
ALL = [(s, c, p) for s in range(3) for c in range(3) for p in range(3)]   # 27 cenas
N_STEPS = 4000
N_SEEDS = 6

ORDER_PT = (0, 1, 2)          # forma, cor, posição
ORDER_X = (2, 0, 1)           # posição, forma, cor (outro idioma)

SHAPE_W = ["barra", "ponto", "anel"]
COLOR_W = ["vermelho", "verde", "azul"]
POS_W = ["topo", "meio", "base"]
WORDS = [SHAPE_W, COLOR_W, POS_W]


def teach(order, train, n=N_STEPS, seed=1):
    lang = make_rich_language(order, K)
    ag = RichSentenceLearner(K, seed=seed)
    rng = np.random.default_rng(0)
    for _ in range(n):
        v = train[int(rng.integers(len(train)))]
        ag.hear(lang(v), v)
    return ag, lang


def render(sentence):
    """Traduz a frase (ids de palavra) para texto legível."""
    out = []
    off = [0, 3, 6]
    for w in sentence:
        for a in range(3):
            if off[a] <= w < off[a] + K[a]:
                out.append(WORDS[a][w - off[a]]); break
    return " ".join(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- generalização: treina em 18 de 27, testa nas 9 nunca vistas (média de sementes) ---
    pt_tr, pt_te, x_tr, x_te = [], [], [], []
    for sd in range(N_SEEDS):
        rng = np.random.default_rng(sd + 3)
        idx = rng.permutation(len(ALL))
        train = [ALL[i] for i in idx[:18]]; test = [ALL[i] for i in idx[18:]]
        ag, lang = teach(ORDER_PT, train, seed=sd + 1)
        pt_tr.append(100 * np.mean([ag.say(v) == lang(v) for v in train]))
        pt_te.append(100 * np.mean([ag.say(v) == lang(v) for v in test]))
        agx, langx = teach(ORDER_X, train, seed=sd + 1)
        x_tr.append(100 * np.mean([agx.say(v) == langx(v) for v in train]))
        x_te.append(100 * np.mean([agx.say(v) == langx(v) for v in test]))
    PT_tr, PT_te = np.mean(pt_tr), np.mean(pt_te)
    X_tr, X_te = np.mean(x_tr), np.mean(x_te)

    # --- baseline ordem aleatória ---
    import itertools
    perms = list(itertools.permutations(range(3)))
    rr = np.random.default_rng(99)
    off = [0, 3, 6]
    def lang_rand(v):
        o = perms[int(rr.integers(len(perms)))]
        return [off[a] + v[a] for a in o]
    agr = RichSentenceLearner(K, seed=1); r = np.random.default_rng(0)
    for _ in range(N_STEPS):
        v = ALL[int(r.integers(len(ALL)))]; agr.hear(lang_rand(v), v)
    und_rand = 100 * np.mean([agr.understand(lang_rand(v)) == v for v in ALL for _ in range(3)])

    # --- curva: generalização vs nº de cenas no treino ---
    sizes = [6, 9, 12, 15, 18, 21, 24]
    gen_curve = []
    for ntr in sizes:
        accs = []
        for sd in range(N_SEEDS):
            rng = np.random.default_rng(sd + 30)
            idx = rng.permutation(len(ALL))
            train = [ALL[i] for i in idx[:ntr]]; test = [ALL[i] for i in idx[ntr:]]
            if not test:
                accs.append(100.0); continue
            ag, lang = teach(ORDER_PT, train, seed=sd + 1)
            accs.append(100 * np.mean([ag.say(v) == lang(v) for v in test]))
        gen_curve.append(np.mean(accs))

    # --- exemplos ---
    ag_ex, lang_ex = teach(ORDER_PT, ALL, seed=1)
    agx_ex, langx_ex = teach(ORDER_X, ALL, seed=1)

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) generalização: treino vs nunca vistas (PT e outro idioma).
    ax = axes[0, 0]
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, [PT_tr, PT_te], w, color="#1f77b4", label="PT (forma,cor,posição)")
    ax.bar(x + w / 2, [X_tr, X_te], w, color="#2ca02c", label="outro idioma (posição,forma,cor)")
    ax.set_xticks(x); ax.set_xticklabels(["cenas de treino\n(18 de 27)", "cenas NUNCA vistas\n(9 de 27)"])
    ax.set_ylabel("frase certa %"); ax.set_ylim(0, 109)
    ax.set_title("(a) Monta descrições de 3 atributos — e generaliza")
    ax.legend(loc="lower left", fontsize=8)
    for xi, v in zip([x[0] - w / 2, x[1] - w / 2], [PT_tr, PT_te]):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    for xi, v in zip([x[0] + w / 2, x[1] + w / 2], [X_tr, X_te]):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)

    # (b) curva: quanto basta ver para compor o resto.
    ax = axes[0, 1]
    ax.plot(sizes, gen_curve, lw=2.2, marker="o", color="#8a3ffc")
    ax.axhline(100.0 / 27, color="#999", ls=":", lw=1, label="acaso (1/27)")
    ax.set_title("(b) Quanto basta ouvir para compor as cenas nunca vistas")
    ax.set_xlabel("nº de cenas vistas no treino (de 27)")
    ax.set_ylabel("frase certa nas NUNCA vistas %"); ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8.5)

    # (c) exemplos concretos.
    ax = axes[1, 0]; ax.axis("off")
    lines = ["(c) Descrições de 3 atributos (vê a cena -> diz a frase):", ""]
    for v in [(0, 0, 0), (1, 2, 1), (2, 1, 2)]:
        cena = f"[{SHAPE_W[v[0]]}, {COLOR_W[v[1]]}, {POS_W[v[2]]}]"
        lines.append(f"cena: {cena}")
        lines.append(f'   PT     -> "{render(ag_ex.say(v))}"')
        lines.append(f'   outro  -> "{render(agx_ex.say(v))}"')
        lines.append("")
    ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=8.8, family="monospace")

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M32 — frases mais ricas (descrição de 3 atributos)\n\n"
            "O agente percebe 3 atributos (forma × cor × posição) e\n"
            "monta/entende uma frase de 3 palavras, descobrindo só\n"
            "ouvindo o grounding e a ORDEM. 27 cenas possíveis.\n\n"
            f"• PT:     treino {PT_tr:.0f}%  |  NUNCA vistas {PT_te:.0f}%\n"
            f"• outro idioma: treino {X_tr:.0f}%  |  NUNCA vistas {X_te:.0f}%\n"
            f"• baseline ordem aleatória: {und_rand:.0f}% (não aprende sintaxe)\n\n"
            "Achado (H15): o agente monta descrições COMPOSICIONAIS\n"
            "de 3 atributos e GENERALIZA para a maior parte das cenas\n"
            "que nunca ouviu — a produtividade da linguagem: infinitos\n"
            "significados de poucas palavras. O mesmo mecanismo aprende\n"
            "ordens de idiomas diferentes. É descrição rica, ancorada,\n"
            "aprendida vivendo — não decorar texto.\n\n"
            "Honesto: 3 atributos, 3 valores, frase de tamanho fixo,\n"
            "escala de brinquedo; mecanismo da descrição composicional,\n"
            "não fluência, recursão nem tamanho variável.",
            transform=ax.transAxes, va="top", fontsize=8.2, family="monospace")

    fig.suptitle("brAIn · M32 — Frases mais ricas: o agente descreve 3 atributos e generaliza para cenas que nunca viu",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m32_rich_sentence.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"PT: treino {PT_tr:.0f}% | nunca vistas {PT_te:.0f}%")
    print(f"Outro idioma: treino {X_tr:.0f}% | nunca vistas {X_te:.0f}%")
    print(f"Baseline ordem aleatoria: entender {und_rand:.0f}%")
    print(f"Exemplo PT: \"{render(ag_ex.say((0,0,0)))}\"  | outro: \"{render(agx_ex.say((0,0,0)))}\"")


if __name__ == "__main__":
    main()
