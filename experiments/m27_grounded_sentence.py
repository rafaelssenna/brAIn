"""M27 — A primeira FRASE ancorada, aprendida ouvindo (PT e EN).

O brAIn grudava PALAVRAS soltas (M21). Falar é montar FRASES. Aqui o agente PERCEBE uma
cena de dois atributos (uma FORMA × uma POSIÇÃO, em pixels), forma seus conceitos sozinho
(predictive coding, M4) e aprende, SÓ OUVINDO, a produzir e entender uma frase de duas
palavras que a descreve — descobrindo o grounding (palavra↔atributo) E a sintaxe (qual
posição da frase carrega qual papel).

O ponto central do foco "qualquer idioma": o MESMO mecanismo aprende ordens diferentes —
  • Português: [forma] [posição]  ("barra topo")
  • Inglês:    [posição] [forma]  ("top bar")
Só muda o idioma ouvido, não o código. Ele aprende a ordem de cada língua vivendo nela.

Provas (método do projeto):
  • produz/entende a frase certa em PT e EN;
  • GENERALIZA para combinações forma×posição NUNCA vistas (produtividade da sintaxe);
  • BASELINE de ordem aleatória (sem regra fixa) -> não aprende a sintaxe (controle);
  • a percepção é REAL: o agente forma seus conceitos de forma/posição dos PIXELS.

Uso:   python experiments/m27_grounded_sentence.py
Salva: experiments/output/m27_grounded_sentence.png

Honesto: escala de brinquedo (frase de 2 palavras, poucos atributos, tamanho fixo). É o
MECANISMO da primeira frase/sintaxe ancorada, como uma criança — NÃO fluência, NÃO recursão.

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
from brain import PredictiveCoder, GroundedSentenceLearner, make_language, PT_ORDER, EN_ORDER  # noqa: E402

np.seterr(all="ignore")     # warnings espúrios de matmul (NumPy/BLAS); resultados são finitos

SEED = 7
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIDE = 8
D = SIDE * SIDE
K_SHAPE = 3                  # 3 formas:  barra horizontal, barra vertical, diagonal
K_POS = 3                   # 3 posições: topo, meio, base
NOISE = 0.05
N_PERC = 2500               # passos para a percepção formar conceitos
N_LANG = 2500               # passos ouvindo o idioma

SHAPE_WORDS = ["barraH", "barraV", "diag"]
POS_WORDS_PT = ["topo", "meio", "base"]
POS_WORDS_EN = ["top", "mid", "bottom"]


def scene(shape: int, pos: int, rng=None):
    """Renderiza uma cena: uma FORMA desenhada numa POSIÇÃO (faixa de linhas) do grid."""
    g = np.zeros((SIDE, SIDE))
    band = {0: range(0, 3), 1: range(3, 5), 2: range(5, 8)}[pos]   # topo/meio/base
    if shape == 0:                                                 # barra horizontal
        r = list(band)[len(list(band)) // 2]; g[r, :] = 1.0
    elif shape == 1:                                               # barra vertical (curta, na faixa)
        for r in band:
            g[r, SIDE // 2] = 1.0
    else:                                                          # diagonal (na faixa)
        for k, r in enumerate(band):
            g[r, min(k, SIDE - 1)] = 1.0
    v = g.ravel()
    if rng is not None:
        v = v + NOISE * rng.standard_normal(D)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def shape_render(val, rng):
    """Canal que isola a FORMA (posição fixa no meio): só a forma varia."""
    return scene(val, 1, rng)


def pos_render(val, rng):
    """Canal que isola a POSIÇÃO (forma fixa = barra H): só a posição varia."""
    return scene(0, val, rng)


def build_perception(render_fn, k, rng_seed):
    """Um predictive coder aprende a categorizar um canal (forma OU posição) dos pixels.
    Devolve uma função pixels->valor-percebido e a discriminabilidade (atenção conjunta)."""
    pc = PredictiveCoder(n_obs=D, n_latent=8, n_infer=30, eta_r=0.1, eta_w=0.05,
                         l2_prior=0.1, seed=rng_seed)
    rng = np.random.default_rng(rng_seed)
    for _ in range(N_PERC):
        pc.learn(render_fn(int(rng.integers(k)), rng))
    # mapeia conceito latente -> valor verdadeiro por votação (atenção conjunta)
    votes = np.zeros((8, k))
    for _ in range(400):
        val = int(rng.integers(k))
        c = int(np.argmax(pc.infer(render_fn(val, rng))))
        votes[c, val] += 1
    concept_to_val = {c: int(np.argmax(votes[c])) for c in range(8) if votes[c].sum() > 0}

    def perceive(x):
        c = int(np.argmax(pc.infer(x)))
        return concept_to_val.get(c, 0)
    discrim = len(set(concept_to_val.values())) / k
    return perceive, discrim


def teach_language(order, perceive_shape, perceive_pos, combos, n_steps):
    """Vive ouvindo um idioma (ordem dada), percebendo as cenas. Devolve o aprendiz e curvas."""
    lang = make_language(order, K_SHAPE)
    ag = GroundedSentenceLearner(K_SHAPE, K_POS, seed=1)
    rng = np.random.default_rng(SEED)
    xs, prod, und = [], [], []
    for t in range(n_steps):
        s, p = combos[int(rng.integers(len(combos)))]
        # PERCEBE a cena (forma e posição) dos pixels — conceitos próprios
        s_perc = perceive_shape(scene(s, 1, rng))      # canal forma
        p_perc = perceive_pos(scene(0, p, rng))        # canal posição
        ag.hear(lang(s, p), s_perc, p_perc)            # ouve a frase do idioma, atenção conjunta
        if t % 100 == 0:
            xs.append(t)
            prod.append(100 * np.mean([ag.say(a, b) == lang(a, b) for (a, b) in combos]))
            und.append(100 * np.mean([ag.understand(lang(a, b)) == (a, b) for (a, b) in combos]))
    return ag, np.array(xs), prod, und


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- percepção REAL: o agente forma conceitos de forma e de posição dos pixels ---
    perceive_shape, disc_shape = build_perception(shape_render, K_SHAPE, rng_seed=1)
    perceive_pos, disc_pos = build_perception(pos_render, K_POS, rng_seed=2)

    all_combos = [(s, p) for s in range(K_SHAPE) for p in range(K_POS)]

    # --- aprende PT e EN com o MESMO mecanismo ---
    ag_pt, xs, prod_pt, und_pt = teach_language(PT_ORDER, perceive_shape, perceive_pos, all_combos, N_LANG)
    ag_en, _, prod_en, und_en = teach_language(EN_ORDER, perceive_shape, perceive_pos, all_combos, N_LANG)

    # --- generalização composicional: treina em 6 de 9, testa nas 3 nunca vistas ---
    rng = np.random.default_rng(3)
    idx = rng.permutation(len(all_combos))
    train = [all_combos[i] for i in idx[:6]]
    test = [all_combos[i] for i in idx[6:]]
    ag_gen, _, _, _ = teach_language(PT_ORDER, perceive_shape, perceive_pos, train, N_LANG)
    lang_pt = make_language(PT_ORDER, K_SHAPE)
    gen_train = 100 * np.mean([ag_gen.say(s, p) == lang_pt(s, p) for (s, p) in train])
    gen_test = 100 * np.mean([ag_gen.say(s, p) == lang_pt(s, p) for (s, p) in test])

    # --- baseline ordem ALEATÓRIA (sem regra fixa) ---
    def random_order_lang():
        rr = np.random.default_rng(99)
        def to_sentence(s, p):
            order = [PT_ORDER, EN_ORDER][int(rr.integers(2))]
            w = {"shape": s, "pos": K_SHAPE + p}
            return [w[role] for role in order]
        return to_sentence
    lang_rand = random_order_lang()
    ag_rand = GroundedSentenceLearner(K_SHAPE, K_POS, seed=1)
    rr = np.random.default_rng(0)
    for _ in range(N_LANG):
        s = int(rr.integers(K_SHAPE)); p = int(rr.integers(K_POS))
        ag_rand.hear(lang_rand(s, p), s, p)
    und_rand = 100 * np.mean([ag_rand.understand(lang_rand(s, p)) == (s, p)
                              for s in range(K_SHAPE) for p in range(K_POS) for _ in range(5)])

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) o mesmo mecanismo aprende a frase em PT e em EN (produzir).
    ax = axes[0, 0]
    ax.plot(xs, prod_pt, lw=2.2, color="#2ca02c", label=f"PT [forma+posição]: {prod_pt[-1]:.0f}%")
    ax.plot(xs, prod_en, lw=2.2, color="#1f77b4", label=f"EN [posição+forma]: {prod_en[-1]:.0f}%")
    ax.set_title("(a) O MESMO mecanismo aprende a frase em PT e EM EN (ordens diferentes)")
    ax.set_xlabel("frases ouvidas"); ax.set_ylabel("produzir frase certa %"); ax.set_ylim(-3, 105)
    ax.legend(loc="lower right", fontsize=9)

    # (b) entender (ouve a frase, identifica a cena) + baseline ordem aleatória.
    ax = axes[0, 1]
    ax.plot(xs, und_pt, lw=2.2, color="#2ca02c", label=f"PT entender: {und_pt[-1]:.0f}%")
    ax.plot(xs, und_en, lw=2.2, color="#1f77b4", label=f"EN entender: {und_en[-1]:.0f}%")
    ax.axhline(und_rand, color="#d62728", ls="--", lw=1.6, label=f"ordem ALEATÓRIA: {und_rand:.0f}%")
    ax.set_title("(b) Entende a frase — mas só quando o idioma TEM uma ordem fixa")
    ax.set_xlabel("frases ouvidas"); ax.set_ylabel("entender a cena %"); ax.set_ylim(-3, 105)
    ax.legend(loc="lower right", fontsize=8.5)

    # (c) generalização composicional: frases NUNCA vistas.
    ax = axes[1, 0]
    bars = ax.bar(["combinações\nde treino", "combinações\nNUNCA vistas"],
                  [gen_train, gen_test], color=["#1f77b4", "#2ca02c"])
    ax.set_ylabel("produzir frase certa %"); ax.set_ylim(0, 109)
    ax.set_title("(c) GENERALIZA: monta frases certas de cenas nunca vistas")
    for b, v in zip(bars, [gen_train, gen_test]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=10)

    # (d) resumo + exemplos concretos.
    ax = axes[1, 1]; ax.axis("off")
    def render_pt(sent):
        return " ".join(SHAPE_WORDS[w] if w < K_SHAPE else POS_WORDS_PT[w - K_SHAPE] for w in sent)
    def render_en(sent):
        return " ".join(SHAPE_WORDS[w] if w < K_SHAPE else POS_WORDS_EN[w - K_SHAPE] for w in sent)
    ex_pt = render_pt(ag_pt.say(0, 0))
    ex_en = render_en(ag_en.say(0, 0))
    ax.text(0.0, 0.98,
            "M27 — a primeira FRASE ancorada, aprendida ouvindo\n\n"
            "O agente PERCEBE a cena (forma × posição) dos pixels,\n"
            "forma seus conceitos (M4) e aprende a FRASE só ouvindo:\n"
            "descobre o grounding (palavra↔atributo) E a ordem do idioma.\n\n"
            f"• percepção:   distingue formas {disc_shape*100:.0f}%, posições {disc_pos*100:.0f}%\n"
            f"• PT produzir/entender: {prod_pt[-1]:.0f}% / {und_pt[-1]:.0f}%\n"
            f"• EN produzir/entender: {prod_en[-1]:.0f}% / {und_en[-1]:.0f}%\n"
            f"• generaliza (nunca visto): {gen_test:.0f}%\n"
            f"• baseline ordem aleatória: {und_rand:.0f}% (não aprende sintaxe)\n\n"
            "Exemplo (vê barra horizontal no topo):\n"
            f"  PT -> \"{ex_pt}\"\n"
            f"  EN -> \"{ex_en}\"\n\n"
            "Achado: o MESMO cérebro aprende a ordem de idiomas\n"
            "diferentes só vivendo neles, e GENERALIZA para frases\n"
            "novas. É a primeira frase com estrutura, ancorada no que\n"
            "percebe — falar começando, não decorar texto.\n\n"
            "Honesto: frase de 2 palavras, escala de brinquedo; é o\n"
            "MECANISMO da sintaxe ancorada, não fluência nem recursão.",
            transform=ax.transAxes, va="top", fontsize=8.2, family="monospace")

    fig.suptitle("brAIn · M27 — A primeira frase ancorada: o mesmo cérebro aprende a falar em português E inglês, ouvindo",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m27_grounded_sentence.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Percepcao: formas {disc_shape*100:.0f}%, posicoes {disc_pos*100:.0f}%")
    print(f"PT: produzir {prod_pt[-1]:.0f}% entender {und_pt[-1]:.0f}% | ordem aprendida {ag_pt.learned_order()}")
    print(f"EN: produzir {prod_en[-1]:.0f}% entender {und_en[-1]:.0f}% | ordem aprendida {ag_en.learned_order()}")
    print(f"Generaliza: treino {gen_train:.0f}% | nunca vistas {gen_test:.0f}%")
    print(f"Baseline ordem aleatoria: entender {und_rand:.0f}%")
    print(f"Exemplo PT: \"{ex_pt}\" | EN: \"{ex_en}\"")


if __name__ == "__main__":
    main()
