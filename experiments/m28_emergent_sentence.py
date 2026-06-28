"""M28 — Conversa emergente com FRASES (do descrever ao conversar).

O M27 aprendeu a primeira frase OUVINDO um professor (que já sabia a resposta). Conversar
é dois cérebros se entendendo SEM professor — coordenando do zero, como dois bebês inventando
uma língua. Aqui dois agentes inventam, juntos, uma mini-língua com FRASES de duas palavras
(forma × posição) só para se entenderem sobre cenas: o falante descreve, o ouvinte adivinha;
acertaram -> reforçam; erraram -> enfraquecem. Ninguém programa o vocabulário NEM a ordem.

Pergunta científica (H11): quando dois cérebros PRECISAM se coordenar, emerge uma língua com
FRASES — vocabulário E uma ORDEM de palavras consistente entre eles? E o código composicional
GENERALIZA para cenas que eles nunca disseram um ao outro?

Provas (método do projeto):
  • a comunicação EMERGE do acaso (sem professor);
  • COMPOSICIONAL generaliza para cenas nunca ditas; HOLÍSTICO (decora) não — o contraste
    mostra que a ESTRUTURA é o que dá produtividade;
  • a ORDEM de palavras emerge CONSISTENTE entre os dois agentes (uma "gramática" comum),
    e é arbitrária entre execuções (como línguas humanas diferem, mas cada uma é coerente).

Uso:   python experiments/m28_emergent_sentence.py
Salva: experiments/output/m28_emergent_sentence.png

Honesto: escala de brinquedo (frase de 2 palavras, poucos atributos, tamanho fixo). É o
MECANISMO da conversa composicional emergente, NÃO linguagem plena (sem recursão, sem
tamanho variável) — horizonte distante, provável híbrido.

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
from brain import EmergentSentenceGame  # noqa: E402

np.seterr(all="ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
K_SHAPE = 3
K_POS = 3
ALL = [(s, p) for s in range(K_SHAPE) for p in range(K_POS)]
N_STEPS = 9000
N_SEEDS = 8


def run(mode, train, eval_scenes, seed, track=False):
    """Dois agentes conversam (sem professor) sobre as cenas de treino. Devolve acurácia
    final no conjunto de avaliação e, se track, a curva de emergência."""
    g = EmergentSentenceGame(K_SHAPE, K_POS, mode=mode, seed=seed)
    rng = np.random.default_rng(seed + 100)
    xs, acc = [], []
    for t in range(N_STEPS):
        s, p = train[int(rng.integers(len(train)))]
        g.play(s, p, speaker_is_A=(t % 2 == 0))
        if track and t % 150 == 0:
            xs.append(t); acc.append(100 * g.accuracy(train))
    return g, np.array(xs), acc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- (1) emergência: a comunicação sobe do acaso, sem professor (treina em tudo) ---
    g_emerge, xs, acc_curve = run("compositional", ALL, ALL, seed=1, track=True)
    chance = 100.0 / len(ALL)

    # --- (2) generalização: treina em 6, testa nas 3 NUNCA ditas; comp vs holístico ---
    comp_train, comp_test, holo_train, holo_test = [], [], [], []
    orders = []
    for sd in range(N_SEEDS):
        rng = np.random.default_rng(sd + 3)
        idx = rng.permutation(len(ALL))
        train = [ALL[i] for i in idx[:6]]; test = [ALL[i] for i in idx[6:]]
        gc, _, _ = run("compositional", train, test, seed=sd + 1)
        gh, _, _ = run("holistic", train, test, seed=sd + 1)
        comp_train.append(100 * gc.accuracy(train)); comp_test.append(100 * gc.accuracy(test))
        holo_train.append(100 * gh.accuracy(train)); holo_test.append(100 * gh.accuracy(test))
        oa, ob = gc.emergent_order()
        orders.append((oa, ob))

    comp_train_m, comp_test_m = np.mean(comp_train), np.mean(comp_test)
    holo_train_m, holo_test_m = np.mean(holo_train), np.mean(holo_test)
    # consistência da ordem entre os DOIS agentes (mesma gramática nos dois?)
    agree = 100 * np.mean([oa == ob for (oa, ob) in orders])
    # quão arbitrária é a ordem entre execuções (varia a "língua")?
    unique_orders = len(set(oa for (oa, ob) in orders))

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) a comunicação EMERGE, sem professor.
    ax = axes[0, 0]
    ax.plot(xs, acc_curve, lw=2.2, color="#2ca02c")
    ax.axhline(chance, color="#bbb", ls="--", lw=1, label=f"acaso ({chance:.0f}%)")
    ax.set_title("(a) A conversa EMERGE sem professor (dois agentes se coordenam)")
    ax.set_xlabel("rodadas de conversa"); ax.set_ylabel("entendimento mútuo %")
    ax.set_ylim(0, 105); ax.legend(loc="lower right", fontsize=9)

    # (b) composicional GENERALIZA; holístico decora.
    ax = axes[0, 1]
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, [comp_train_m, comp_test_m], w, color="#1f77b4", label="composicional")
    ax.bar(x + w / 2, [holo_train_m, holo_test_m], w, color="#d62728", label="holístico (decora)")
    ax.set_xticks(x); ax.set_xticklabels(["cenas de treino", "cenas NUNCA ditas"])
    ax.set_ylabel("entendimento mútuo %"); ax.set_ylim(0, 109)
    ax.set_title("(b) Composicional GENERALIZA; holístico só decora")
    ax.legend(loc="upper right", fontsize=8.5)
    for xi, v in zip([x[0] - w / 2, x[1] - w / 2], [comp_train_m, comp_test_m]):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    for xi, v in zip([x[0] + w / 2, x[1] + w / 2], [holo_train_m, holo_test_m]):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)

    # (c) a ORDEM de palavras emerge — consistente entre os dois, arbitrária entre execuções.
    ax = axes[1, 0]
    labels = ["barra+posição\n(shape,pos)", "posição+barra\n(pos,shape)"]
    counts = [sum(1 for (oa, _) in orders if oa == ("shape", "pos")),
              sum(1 for (oa, _) in orders if oa == ("pos", "shape"))]
    ax.bar(labels, counts, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("nº de execuções (sementes)")
    ax.set_title(f"(c) A ORDEM emerge: arbitrária entre execuções, mas os 2 agentes CONCORDAM ({agree:.0f}%)")
    ax.set_ylim(0, N_SEEDS + 0.5)

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M28 — conversa emergente com FRASES\n\n"
            "Dois agentes inventam, do zero e SEM PROFESSOR, uma\n"
            "mini-língua com frases de 2 palavras (forma × posição)\n"
            "só para se entenderem sobre cenas. Acertaram -> reforçam;\n"
            "erraram -> enfraquecem. Vocabulário E ordem EMERGEM.\n\n"
            f"• comunicação emerge:  acaso {chance:.0f}% -> {acc_curve[-1]:.0f}% (sem professor)\n"
            f"• composicional:  treino {comp_train_m:.0f}%  |  NUNCA DITAS {comp_test_m:.0f}%\n"
            f"• holístico:      treino {holo_train_m:.0f}%  |  NUNCA DITAS {holo_test_m:.0f}%\n"
            f"• ordem das palavras: os 2 agentes concordam em {agree:.0f}%\n"
            f"  das execuções ({unique_orders} ordens diferentes entre sementes)\n\n"
            "Achado (H11): quando dois cérebros PRECISAM se coordenar,\n"
            "emerge uma língua com FRASES — vocabulário + uma ORDEM de\n"
            "palavras consistente entre eles. Composicional GENERALIZA\n"
            "para cenas nunca ditas (produtividade); holístico não. A\n"
            "ordem é arbitrária entre execuções (como línguas humanas\n"
            "diferem) mas coerente dentro de cada conversa. É conversar\n"
            "começando, não decorar.\n\n"
            "Honesto: frase de 2 palavras, escala de brinquedo; mecanismo\n"
            "da conversa composicional, não fluência nem recursão.",
            transform=ax.transAxes, va="top", fontsize=8.2, family="monospace")

    fig.suptitle("brAIn · M28 — Conversa emergente: dois cérebros inventam uma língua com frases (e uma gramática) só para se entender",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m28_emergent_sentence.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Emerge: acaso {chance:.0f}% -> {acc_curve[-1]:.0f}% (sem professor)")
    print(f"Composicional: treino {comp_train_m:.0f}% | nunca ditas {comp_test_m:.0f}%")
    print(f"Holistico:     treino {holo_train_m:.0f}% | nunca ditas {holo_test_m:.0f}%")
    print(f"Ordem: 2 agentes concordam em {agree:.0f}% das execucoes; {unique_orders} ordens distintas")


if __name__ == "__main__":
    main()
