"""M25 — O organismo corporificado que fala (a costura que faltava).

Une as duas metades do cérebro que viviam separadas:
  • o agente com CORPO (M7/M13) — navega, percebe, é curioso, lembra — mas mudo;
  • o agente que FALA (M20→M24) — percebe, lembra, fala — mas sem corpo.

Aqui dois organismos COMPLETOS vivem num anel de lugares: cada um tem corpo, navega
por CURIOSIDADE (learning progress, M5), percebe o objeto de onde está (M4/M24),
lembra (M8) e fala (M16/M21). Eles só jogam o jogo de linguagem quando se ENCONTRAM
no mesmo lugar olhando o mesmo objeto — atenção conjunta corporificada.

A diferença crucial para o M20/M21: lá os objetos eram MOSTRADOS (percepção passiva).
Aqui o agente só vê um objeto se FOR ATÉ o lugar dele, e só alinha um símbolo com o
outro se ambos estiverem LÁ juntos. Ver e nomear custam andar.

Pergunta científica (H8): a linguagem ainda emerge quando ver custa ação? E qual é o
custo da corporificação? Baseline: o MESMO organismo SEM corpo (objetos mostrados,
estilo M20) — sem o gargalo do encontro.

Uso:   python experiments/m25_embodied_language.py
Salva: experiments/output/m25_embodied_language.png

Honesto: escala de brinquedo. É o organismo inteiro num laço (perceber+mover+lembrar+
ser curioso+falar), não um cérebro pronto.

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
from brain import (RingWorld, EmbodiedLanguageAgent, LivingAgent,  # noqa: E402
                   communication_accuracy)

np.seterr(all="ignore")     # warnings espúrios de matmul (NumPy/BLAS); resultados são finitos

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GRID = 8
N_LOC = 8
NOISE_LOCS = (3, 6)
N_LATENT = 12
EPS = 0.25
DWELL = 6
N_STEPS = 9000
NOISE = 0.06


def _comm_both(A, B, clean):
    return 50 * (communication_accuracy(A, B, clean) + communication_accuracy(B, A, clean))


def live_embodied(world, learn_locs, clean, n_steps):
    """Dois organismos COM corpo: navegam por curiosidade, se encontram, falam."""
    K = len(learn_locs)
    A = EmbodiedLanguageAgent(world, N_LATENT, K, eps=EPS, dwell=DWELL, seed=1)
    B = EmbodiedLanguageAgent(world, N_LATENT, K, eps=EPS, dwell=DWELL, seed=2)
    rng = np.random.default_rng(SEED)
    xs, comm, surp, enc_cum = [], [], [], []
    encounters = 0
    for t in range(n_steps):
        A.perceive_and_learn(rng); B.perceive_and_learn(rng)
        if A.pos == B.pos and A.pos in learn_locs:        # atenção conjunta corporificada
            encounters += 1
            obj = world.patterns[A.pos]
            spk, lis = (A, B) if t % 2 == 0 else (B, A)
            c = spk.concept_of(obj); m = spk.speak(c)
            reward = (lis.listen(m) == lis.concept_of(obj))
            spk.reinforce_speak(c, m, reward)
            lis.learn_listen(m, lis.concept_of(obj))
        A.navigate(); B.navigate()
        if t % 150 == 0:
            xs.append(t)
            comm.append(_comm_both(A, B, clean))
            surp.append(np.mean([A.surprise_on(clean[j]) for j in range(K)]))
            enc_cum.append(encounters)
    return A, B, np.array(xs), comm, surp, enc_cum, encounters


def live_disembodied(world, learn_locs, clean, n_steps):
    """Baseline SEM corpo: os objetos são mostrados (estilo M20), sem gargalo de encontro."""
    K = len(learn_locs)
    A = LivingAgent(world.D, N_LATENT, K, seed=1)
    B = LivingAgent(world.D, N_LATENT, K, seed=2)
    rng = np.random.default_rng(SEED)
    xs, comm = [], []
    for t in range(n_steps):
        oi = int(rng.integers(K))
        obj = clean[oi] + NOISE * rng.standard_normal(world.D)
        A.learn_perception(obj); B.learn_perception(obj)
        spk, lis = (A, B) if t % 2 == 0 else (B, A)
        c = spk.concept(obj); m = spk.speak(c)
        reward = (lis.listen(m) == lis.concept(obj))
        spk.reinforce_speak(c, m, reward)
        lis.learn_listen(m, lis.concept(obj))
        if t % 150 == 0:
            xs.append(t)
            comm.append(50 * (_living_comm(A, B, clean) + _living_comm(B, A, clean)))
    return np.array(xs), comm


def _living_comm(spk, lis, clean):
    lc = [lis.concept(clean[j]) for j in range(len(clean))]
    tot = 0.0
    for o in range(len(clean)):
        m = spk.speak(spk.concept(clean[o]), explore=False)
        cands = [j for j in range(len(clean)) if lc[j] == lis.listen(m)]
        if o in cands:
            tot += 1.0 / len(cands)
    return tot / len(clean)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    world = RingWorld(n_locations=N_LOC, grid=GRID, noise_locs=NOISE_LOCS, seed=0)
    learn_locs = world.learnable_locs()
    clean = [world.patterns[l] for l in learn_locs]
    K = len(learn_locs)

    print("Vivendo o organismo CORPORIFICADO (corpo + curiosidade + percepcao + linguagem)...")
    A, B, xs, comm_e, surp_e, enc_cum, n_enc = live_embodied(world, learn_locs, clean, N_STEPS)
    print("Vivendo o baseline DESCORPORIFICADO (objetos mostrados, estilo M20)...")
    xd, comm_d = live_disembodied(world, learn_locs, clean, N_STEPS)

    disc = len(set(A.concept_of(clean[j]) for j in range(K))) / K
    chance = 100.0 / K

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) percepção corporificada: a surpresa cai navegando o mundo.
    ax = axes[0, 0]
    ax.plot(xs, surp_e, lw=2.2, color="#2ca02c")
    ax.set_title("(a) Percepção corporificada: a surpresa cai vivendo o mundo")
    ax.set_xlabel("passos de vida"); ax.set_ylabel("erro de previsão (surpresa)")

    # (b) a linguagem emerge COM corpo, comparável ao sem corpo (objetos mostrados).
    ax = axes[0, 1]
    ax.plot(xd, comm_d, lw=2.2, color="#1f77b4", label=f"sem corpo (mostrado): {comm_d[-1]:.0f}%")
    ax.plot(xs, comm_e, lw=2.2, color="#d62728", label=f"com corpo (ir até ver): {comm_e[-1]:.0f}%")
    ax.axhline(chance, color="#bbb", ls="--", lw=1, label=f"acaso ({chance:.0f}%)")
    ax.set_title("(b) A linguagem emerge COM corpo (comparável ao sem corpo)")
    ax.set_xlabel("passos de vida"); ax.set_ylabel("comunicação %"); ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8.5)

    # (c) o gargalo: a comunicação acompanha os ENCONTROS (atenção conjunta corporificada).
    ax = axes[1, 0]
    ax.plot(enc_cum, comm_e, lw=2, color="#8a3ffc", marker="o", ms=2.5)
    ax.set_title("(c) Só se fala do que se atende JUNTO: comunicação × encontros")
    ax.set_xlabel("encontros co-localizados (acumulado)"); ax.set_ylabel("comunicação %")
    ax.set_ylim(0, 105)

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M25 — o organismo corporificado que fala\n\n"
            "Dois organismos COMPLETOS num anel de lugares: cada um\n"
            "tem CORPO, navega por CURIOSIDADE, PERCEBE o objeto de\n"
            "onde está, LEMBRA e FALA. Só alinham a língua quando se\n"
            "ENCONTRAM no mesmo lugar (atenção conjunta corporificada).\n\n"
            f"• percepção:    surpresa {surp_e[0]:.2f} -> {surp_e[-1]:.3f}\n"
            f"• conceitos:    {disc*100:.0f}% dos objetos distinguidos\n"
            f"• encontros:    {n_enc} co-localizados (com objeto)\n"
            f"• comunicação:  com corpo {comm_e[-1]:.0f}%  |  sem corpo {comm_d[-1]:.0f}%\n"
            f"                (acaso {chance:.0f}%)\n\n"
            "Achado (H8): a linguagem EMERGE mesmo quando ver custa\n"
            "andar — gated pela atenção conjunta: só se nomeia o que os\n"
            "dois corpos atenderam JUNTOS, e a comunicação acompanha os\n"
            "encontros (painel c, o sinal mais limpo). É a metade que\n"
            "faltava: perceber + mover + lembrar + ser curioso + falar.\n\n"
            "Honesto: corpo e sem-corpo ficam COMPARÁVEIS (a diferença\n"
            "varia com a semente — não afirmamos que um vence o outro);\n"
            "escala de brinquedo; encontro = co-localização simples; é o\n"
            "MECANISMO do organismo num laço, não um cérebro pronto.",
            transform=ax.transAxes, va="top", fontsize=8.6, family="monospace")

    fig.suptitle("brAIn · M25 — O organismo corporificado que fala: perceber, mover, lembrar, ser curioso e falar num laço só",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m25_embodied_language.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Surpresa:     {surp_e[0]:.2f} -> {surp_e[-1]:.3f}")
    print(f"Conceitos:    {disc*100:.0f}% distinguidos")
    print(f"Encontros:    {n_enc}")
    print(f"Comunicacao:  com corpo {comm_e[-1]:.0f}% | sem corpo {comm_d[-1]:.0f}% (acaso {chance:.0f}%)")


if __name__ == "__main__":
    main()
