"""M21 — Ele aprende palavras de verdade do PORTUGUÊS, ancoradas no que percebe.

Até aqui os agentes inventavam símbolos próprios. Aqui um agente aprende PALAVRAS
REAIS do português, do jeito que uma criança aprende: ele vê o mundo (barras em
posições diferentes), forma seus conceitos sozinho (predictive coding, M4) e um
"professor" diz a palavra certa junto (atenção conjunta / aprendizado
cross-situacional). Ele aprende as duas direções da língua:
  • NOMEAR (produção): vê um objeto e diz a palavra certa.
  • APONTAR (compreensão): ouve uma palavra e aponta o objeto certo.

Tudo online, com a percepção ainda se formando. A pergunta honesta: ele consegue
grudar palavras reais em conceitos que ainda estão nascendo? E até onde? (Resposta:
até onde ele CONSEGUE DISTINGUIR as coisas — só se nomeia o que se percebe.)

Uso:   python experiments/m21_portuguese.py
Salva: experiments/output/m21_portuguese.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LivingAgent  # noqa: E402

SEED = 7
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIDE = 8
D = SIDE * SIDE
N_STEPS = 5000
NOISE = 0.06

# O mundinho dele: 3 barras horizontais e 3 verticais. As palavras de português
# descrevem onde a barra está — o significado é perceptual de verdade.
ROWS = [1, 4, 6]
COLS = [1, 4, 6]
WORDS = ["topo", "meio", "base", "esquerda", "centro", "direita"]


def objects():
    pats = []
    for r in ROWS:
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    for c in COLS:
        p = np.zeros((SIDE, SIDE)); p[:, c] = 1.0; pats.append(p.ravel())
    P = np.array(pats)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def naming_accuracy(agent, P):
    """Vê cada objeto e diz a palavra: fração de nomes corretos."""
    ok = 0
    for o in range(len(P)):
        if agent.speak(agent.concept(P[o]), explore=False) == o:
            ok += 1
    return ok / len(P)


def comprehension_accuracy(agent, P):
    """Ouve cada palavra e aponta o objeto (crédito repartido se houver empate)."""
    cs = [agent.concept(P[j]) for j in range(len(P))]
    tot = 0.0
    for w in range(len(P)):
        cands = [j for j in range(len(P)) if cs[j] == agent.listen(w)]
        if w in cands:
            tot += 1.0 / len(cands)
    return tot / len(P)


def teach(seed, n_latent, n_steps, noise, rng_seed):
    """Um agente vive o mundo enquanto o professor diz as palavras. Devolve as curvas."""
    P = objects()
    K = len(P)
    agent = LivingAgent(D, n_latent, K, seed=seed)
    rng = np.random.default_rng(rng_seed)
    xs, nam, comp, surp = [], [], [], []
    for t in range(n_steps):
        o = int(rng.integers(K))                       # o professor mostra um objeto
        obj = P[o] + noise * rng.standard_normal(D)
        agent.learn_perception(obj)                    # ele aprende a VER
        c = agent.concept(obj)
        agent.learn_listen(o, c)                        # ouve a palavra -> meu conceito
        agent.reinforce_speak(c, o, True)               # meu conceito -> aquela palavra
        if t % 100 == 0:
            xs.append(t)
            nam.append(100 * naming_accuracy(agent, P))
            comp.append(100 * comprehension_accuracy(agent, P))
            surp.append(np.mean([agent.pc.prediction_error(P[j]) for j in range(K)]))
    return agent, np.array(xs), nam, comp, surp


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    P = objects(); K = len(P)

    # --- agente principal: capacidade boa, aprende português direito ---
    agent, xs, nam, comp, surp = teach(seed=1, n_latent=12, n_steps=N_STEPS,
                                       noise=NOISE, rng_seed=SEED)
    pmast = 100.0 * (1.0 - np.array(surp) / surp[0])

    # --- o diálogo: mostro um objeto, ele diz a palavra; falo uma palavra, ele aponta ---
    print("\n--- NOMEAR (vejo um objeto, ele diz a palavra) ---")
    for o in range(K):
        said = WORDS[agent.speak(agent.concept(P[o]), explore=False)]
        mark = "OK" if said == WORDS[o] else "x"
        print(f"  vejo [{WORDS[o]:9s}] -> ele diz: \"{said}\"   {mark}")
    print("\n--- APONTAR (falo uma palavra, ele aponta o objeto) ---")
    cs = [agent.concept(P[j]) for j in range(K)]
    for w in range(K):
        cands = [j for j in range(K) if cs[j] == agent.listen(w)]
        pointed = WORDS[cands[0]] if len(cands) == 1 else "(?)"
        mark = "OK" if (len(cands) == 1 and cands[0] == w) else "x"
        print(f"  ouço \"{WORDS[w]:9s}\" -> ele aponta: [{pointed}]   {mark}")

    # --- o limite honesto: nomear so vai ate onde ele DISTINGUE as coisas ---
    disc, namf = [], []
    for nl in (2, 3, 4, 6, 9, 12):
        for sd in (1, 2):
            ag, _, _, _, _ = teach(seed=10 * sd + nl, n_latent=nl, n_steps=2500,
                                   noise=NOISE, rng_seed=100 + sd)
            cs2 = [ag.concept(P[j]) for j in range(K)]
            disc.append(len(set(cs2)) / K)
            namf.append(naming_accuracy(ag, P))
    disc, namf = np.array(disc), np.array(namf)
    r = float(np.corrcoef(disc, namf)[0, 1])

    # ====================== figura ======================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    # (a) co-emergência: aprende a ver, depois aprende a nomear/compreender.
    ax = axes[0, 0]
    ax.plot(xs, pmast, lw=2.2, color="#2ca02c", label="percepção (surpresa domada)")
    ax.plot(xs, nam, lw=2.2, color="#d62728", label="nomear (produção)")
    ax.plot(xs, comp, lw=2.2, color="#1f77b4", label="apontar (compreensão)")
    ax.set_title("(a) Ele aprende as palavras de PT vivendo (a língua vem atrás da percepção)")
    ax.set_xlabel("passos de vida"); ax.set_ylabel("%"); ax.set_ylim(-2, 105)
    ax.legend(loc="lower right", fontsize=8.5)

    # (b) o léxico aprendido: para cada objeto, a palavra que ele diz.
    ax = axes[0, 1]
    lex = np.array([agent._sm(agent.S[agent.concept(P[o])]) for o in range(K)])
    ax.imshow(lex, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(K)); ax.set_xticklabels(WORDS, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels(WORDS, fontsize=8)
    ax.set_xlabel("palavra que ele DIZ"); ax.set_ylabel("objeto que ele VÊ")
    ax.set_title("(b) Léxico aprendido (diagonal = nomeou certo)")

    # (c) o limite honesto: nomear <= o que ele distingue.
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1, label="teto (y=x)")
    ax.scatter(disc, namf, s=46, color="#8a3ffc", alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("percepção: fração de objetos que ele DISTINGUE")
    ax.set_ylabel("acurácia ao NOMEAR")
    ax.set_title(f"(c) Só nomeia o que percebe (r = {r:.2f}, sob o teto)")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.legend(loc="upper left", fontsize=8)

    # (d) resumo.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.99,
            "M21 — aprendendo PORTUGUÊS de verdade (ancorado na percepção)\n\n"
            "Um agente vê o mundo, forma seus conceitos sozinho (M4) e um\n"
            "professor diz a palavra real junto (atenção conjunta). Ele\n"
            "aprende as DUAS direções da língua:\n"
            "  • NOMEAR: vê o objeto e diz a palavra de português.\n"
            "  • APONTAR: ouve a palavra e aponta o objeto certo.\n\n"
            f"• nomear (final):       {nam[-1]:.0f}%\n"
            f"• apontar (final):      {comp[-1]:.0f}%\n"
            f"• palavras aprendidas:  {', '.join(WORDS)}\n\n"
            "Achado: a língua fica ATRÁS da percepção e gruda quando os\n"
            "conceitos assentam. E só nomeia o que CONSEGUE DISTINGUIR\n"
            f"(painel c): nomear acompanha a percepção, r = {r:.2f}, sob o teto.\n\n"
            "Honesto: 6 palavras, mundo de brinquedo. É o MECANISMO de\n"
            "aprender palavras como criança (grounding), não fluência —\n"
            "português pleno pede outra escala (ou um híbrido).",
            transform=ax.transAxes, va="top", fontsize=8.8, family="monospace")

    fig.suptitle("brAIn · M21 — Ele aprende palavras de português ancoradas no que percebe",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m21_portuguese.png")
    fig.savefig(out, dpi=120)
    print(f"\nFigura salva em: {out}")
    print(f"Nomear: {nam[0]:.0f}% -> {nam[-1]:.0f}%   Apontar: {comp[0]:.0f}% -> {comp[-1]:.0f}%   (r percepção-nome = {r:.2f})")


if __name__ == "__main__":
    main()
