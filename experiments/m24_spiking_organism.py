"""M24 — O organismo vivo PERCEBENDO com neurônios que DISPARAM (spiking).

A maior lacuna honesta do projeto (marcada com ⚠️ no README): o organismo vivo do
M20 (dois agentes que percebem, lembram e falam) sempre percebeu com um substrato
RATE-BASED (uma taxa contínua). Mas o cérebro não funciona assim — neurônios DISPARAM
(spikes), eventos discretos no tempo. O M10 já tinha esse substrato spiking (neurônios
LIF reais), mas vivia ISOLADO, fora do organismo. Aqui ele entra no laço da vida.

Pergunta científica: a cognição (percepção + linguagem co-emergindo) SOBREVIVE quando
a percepção é feita por neurônios que de fato disparam? E qual é o custo?

Roda o MESMO organismo do M20, lado a lado, em dois substratos:
  • RATE-BASED (M20 original, taxa contínua);
  • SPIKING (M10 dentro do organismo, neurônios LIF que disparam) — M24.

Uso:   python experiments/m24_spiking_organism.py
Salva: experiments/output/m24_spiking_organism.png

Honesto: escala de brinquedo (6 objetos, D=64); o laço spiking é LENTO em Python
(o M6/M22 já mostraram que a moeda é OPERAÇÃO SINÁPTICA, não o relógio). Isto prova
que a cognição roda sobre spikes reais em miniatura, NÃO escala nem eficiência cerebral.

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
from brain import LivingAgent  # noqa: E402

np.seterr(all="ignore")     # warnings espúrios de matmul (NumPy 2.0/BLAS) — resultados são finitos

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIDE = 8
D = SIDE * SIDE
K = 6
N_LATENT = 16
N_SYMBOLS = 8
N_STEPS_RATE = 6000        # rate-based é rápido
N_STEPS_SPIKE = 2000       # spiking é lento (~80ms/passo); menos passos, mesma história
NOISE = 0.06


def objects():
    pats = []
    for r in range(0, SIDE, 2):
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    for c in range(0, SIDE, 3):
        p = np.zeros((SIDE, SIDE)); p[:, c] = 1.0; pats.append(p.ravel())
    P = np.array(pats[:K])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def comm_acc(spk, lis, P):
    lc = [lis.concept(P[j]) for j in range(len(P))]
    tot = 0.0
    for o in range(len(P)):
        m = spk.speak(spk.concept(P[o]), explore=False)
        cands = [j for j in range(len(P)) if lc[j] == lis.listen(m)]
        if o in cands:
            tot += 1.0 / len(cands)
    return tot / len(P)


def live(spiking, n_steps, P, sample_every):
    """Roda o organismo do M20 (dois agentes) e devolve as curvas de co-emergência."""
    A = LivingAgent(D, N_LATENT, N_SYMBOLS, seed=1, spiking=spiking)
    B = LivingAgent(D, N_LATENT, N_SYMBOLS, seed=2, spiking=spiking)
    rng = np.random.default_rng(SEED)
    xs, comm, surp = [], [], []
    for t in range(n_steps):
        oi = int(rng.integers(K))
        obj = P[oi] + NOISE * rng.standard_normal(D)
        A.learn_perception(obj); B.learn_perception(obj)
        spk, lis = (A, B) if t % 2 == 0 else (B, A)
        c = spk.concept(obj); m = spk.speak(c)
        lc = [lis.concept(P[j]) for j in range(K)]
        cands = [j for j in range(K) if lc[j] == lis.listen(m)]
        guess = int(rng.choice(cands)) if cands else int(rng.integers(K))
        spk.reinforce_speak(c, m, guess == oi)
        lis.learn_listen(m, lis.concept(obj))
        if t % sample_every == 0:
            xs.append(t)
            comm.append(100 * 0.5 * (comm_acc(A, B, P) + comm_acc(B, A, P)))
            surp.append(np.mean([A.pc.prediction_error(P[j]) for j in range(K)]))
    return A, np.array(xs), comm, surp


def active_frac(agent, P):
    return float(np.mean([agent.pc.active_fraction(P[o]) for o in range(len(P))]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    P = objects()

    print("Rodando o organismo RATE-BASED (M20 original)...")
    A_rate, xr, comm_r, surp_r = live(False, N_STEPS_RATE, P, sample_every=100)
    print("Rodando o organismo SPIKING (neuronios LIF que disparam)... (lento)")
    A_spk, xs, comm_s, surp_s = live(True, N_STEPS_SPIKE, P, sample_every=50)

    disc_r = A_rate.discriminability(P)
    disc_s = A_spk.discriminability(P)
    af_r, af_s = active_frac(A_rate, P), active_frac(A_spk, P)

    # raster: os neurônios LIF do organismo spiking DISPARANDO ao perceber um objeto
    raster = A_spk.pc.spike_raster(P[0])

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) percepção (surpresa) cai nos dois substratos.
    ax = axes[0, 0]
    ax.plot(xr, surp_r, lw=2, color="#1f77b4", label="rate-based (M20)")
    ax.plot(xs, surp_s, lw=2, color="#d62728", label="spiking (LIF, M24)")
    ax.set_title("(a) Percepção: a surpresa cai mesmo percebendo com spikes")
    ax.set_xlabel("passos de vida"); ax.set_ylabel("erro de previsão (surpresa)")
    ax.legend(loc="upper right", fontsize=9)

    # (b) linguagem co-emerge nos dois substratos.
    ax = axes[0, 1]
    ax.plot(xr, comm_r, lw=2.2, color="#1f77b4", label=f"rate: {comm_r[-1]:.0f}%")
    ax.plot(xs, comm_s, lw=2.2, color="#d62728", label=f"spiking: {comm_s[-1]:.0f}%")
    ax.set_title("(b) A linguagem co-emerge sobre neurônios que disparam")
    ax.set_xlabel("passos de vida"); ax.set_ylabel("comunicação %"); ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)

    # (c) o substrato de verdade: raster dos neurônios LIF disparando ao perceber.
    ax = axes[1, 0]
    for i, times in enumerate(raster):
        if len(times):
            ax.scatter(times, np.full_like(times, i), s=10, color="#d62728")
    ax.set_title("(c) O substrato spiking: neurônios LIF disparando ao perceber")
    ax.set_xlabel("tempo (ms)"); ax.set_ylabel("neurônio latente")
    ax.set_ylim(-1, N_LATENT)

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M24 — o organismo vivo percebendo com SPIKES\n\n"
            "O MESMO organismo do M20 (perceber + lembrar +\n"
            "falar, dois agentes), mas a percepção agora vem de\n"
            "neurônios LIF REAIS que DISPARAM (substrato do M10\n"
            "dentro do organismo). Fecha a lacuna spiking↔cognição.\n\n"
            f"• surpresa:     rate {surp_r[-1]:.3f}  |  spiking {surp_s[-1]:.3f}\n"
            f"• comunicação:  rate {comm_r[-1]:.0f}%  |  spiking {comm_s[-1]:.0f}%\n"
            f"• conceitos:    rate {disc_r*100:.0f}%  |  spiking {disc_s*100:.0f}% distinguidos\n"
            f"• atividade:    rate {af_r*100:.0f}%  |  spiking {af_s*100:.0f}% (esparsidade\n"
            "                emergente do limiar do spike)\n\n"
            "Achado: a cognição (percepção + linguagem) SOBREVIVE\n"
            "quando a percepção é feita por spikes reais. O passo\n"
            "mais 'cérebro de verdade' do organismo até aqui.\n\n"
            "Honesto: escala de brinquedo; o laço spiking é lento\n"
            "(moeda = operação sináptica, não relógio); spiking\n"
            "rate-coded (taxa de disparo), não código temporal puro.",
            transform=ax.transAxes, va="top", fontsize=8.8, family="monospace")

    fig.suptitle("brAIn · M24 — O organismo vivo percebe com neurônios que disparam: a cognição sobrevive aos spikes",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m24_spiking_organism.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Surpresa:     rate {surp_r[-1]:.3f} | spiking {surp_s[-1]:.3f}")
    print(f"Comunicacao:  rate {comm_r[-1]:.0f}% | spiking {comm_s[-1]:.0f}%")
    print(f"Conceitos:    rate {disc_r*100:.0f}% | spiking {disc_s*100:.0f}%")
    print(f"Atividade:    rate {af_r*100:.1f}% | spiking {af_s*100:.1f}%")


if __name__ == "__main__":
    main()
