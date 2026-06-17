"""M23 — O organismo vivo rodando no substrato EFICIENTE (esparso).

O M20 mostrou percepção e linguagem co-emergindo num laço só. O M22 mostrou que um
código esparso (cortical, ~10% ativo) faz o mesmo gastando muito menos operações
sinápticas. Este marco junta os dois: o MESMO organismo do M20 (dois agentes que
percebem, lembram e falam) rodando sobre o substrato esparso, lado a lado com o denso.

A pergunta honesta: a cognição (a co-emergência percepção↔linguagem) SOBREVIVE à
esparsidade? E quanto de energia se economiza? É o passo que costura a fratura entre
o substrato eficiente e o organismo integrado.

Uso:   python experiments/m23_efficient_organism.py
Salva: experiments/output/m23_efficient_organism.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LivingAgent, OpCounter  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIDE = 8
D = SIDE * SIDE
K = 6
N_LATENT = 16
N_SYMBOLS = 8
N_STEPS = 6000
NOISE = 0.06
K_SPARSE = 2                 # k-WTA: 2/16 = 12.5% ativo (cortical)


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


def live(sparse_k, P):
    """Roda o organismo do M20 (dois agentes) e devolve as curvas de co-emergência."""
    A = LivingAgent(D, N_LATENT, N_SYMBOLS, seed=1, sparse_k=sparse_k)
    B = LivingAgent(D, N_LATENT, N_SYMBOLS, seed=2, sparse_k=sparse_k)
    rng = np.random.default_rng(SEED)
    xs, comm, surp = [], [], []
    for t in range(N_STEPS):
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
        if t % 100 == 0:
            xs.append(t)
            comm.append(100 * 0.5 * (comm_acc(A, B, P) + comm_acc(B, A, P)))
            surp.append(np.mean([A.pc.prediction_error(P[j]) for j in range(K)]))
    return A, np.array(xs), comm, surp


def synops_per_perception(agent, P):
    """SynOps (dirigido a eventos) de uma percepção média do organismo treinado."""
    cnt = OpCounter()
    for o in range(len(P)):
        agent.pc.infer(P[o], counter=cnt)
    return cnt.total_warm() / len(P)


def active_frac(agent, P):
    return float(np.mean([agent.pc.active_fraction(P[o]) for o in range(len(P))]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    P = objects()

    A_dense, xs, comm_d, surp_d = live(None, P)        # organismo denso (M20 original)
    A_spar, _, comm_s, surp_s = live(K_SPARSE, P)      # organismo esparso (M22 dentro)

    syn_d = synops_per_perception(A_dense, P)
    syn_s = synops_per_perception(A_spar, P)
    factor = syn_d / syn_s
    af_d, af_s = active_frac(A_dense, P), active_frac(A_spar, P)
    disc_d = A_dense.discriminability(P)
    disc_s = A_spar.discriminability(P)

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) comunicação co-emerge nos dois substratos.
    ax = axes[0, 0]
    ax.plot(xs, comm_d, lw=2.2, color="#1f77b4", label=f"denso: {comm_d[-1]:.0f}%")
    ax.plot(xs, comm_s, lw=2.2, color="#2ca02c", label=f"esparso (k={K_SPARSE}): {comm_s[-1]:.0f}%")
    ax.set_title("(a) A linguagem co-emerge IGUAL no substrato esparso")
    ax.set_xlabel("passos de vida"); ax.set_ylabel("comunicação %"); ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)

    # (b) percepção (surpresa) cai nos dois.
    ax = axes[0, 1]
    ax.plot(xs, surp_d, lw=2, color="#1f77b4", label="denso")
    ax.plot(xs, surp_s, lw=2, color="#2ca02c", label="esparso")
    ax.set_title("(b) Percepção: a surpresa cai nos dois")
    ax.set_xlabel("passos de vida"); ax.set_ylabel("erro de previsão (surpresa)")
    ax.legend(loc="upper right", fontsize=9)

    # (c) energia: o organismo esparso gasta menos por percepção.
    ax = axes[1, 0]
    bars = ax.bar(["denso", f"esparso\nk={K_SPARSE}"], [syn_d, syn_s],
                  color=["#1f77b4", "#2ca02c"])
    ax.set_yscale("log"); ax.set_ylabel("SynOps por percepção")
    ax.set_title(f"(c) Energia: o organismo esparso usa ~{factor:.1f}x menos")
    for b, v in zip(bars, [syn_d, syn_s]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M23 — o organismo vivo no substrato eficiente\n\n"
            "O MESMO organismo do M20 (perceber + lembrar +\n"
            "falar, dois agentes), agora rodando no código\n"
            "esparso do M22 (k-WTA, cortical).\n\n"
            f"• comunicação:  denso {comm_d[-1]:.0f}%  ->  esparso {comm_s[-1]:.0f}%\n"
            f"• conceitos:    denso {disc_d*100:.0f}%  ->  esparso {disc_s*100:.0f}% distinguidos\n"
            f"• atividade:    {af_d*100:.0f}%  ->  {af_s*100:.0f}% (cortical)\n"
            f"• ENERGIA:      ~{factor:.1f}x menos SynOps por percepção\n\n"
            "Achado: a cognição (percepção + linguagem) SOBREVIVE\n"
            "à esparsidade — co-emerge igual, gastando menos. A\n"
            "fratura substrato-eficiente <-> organismo fechou.\n\n"
            "Honesto: moeda = operação sináptica, não relógio;\n"
            "escala de brinquedo; princípio, não eficiência cerebral.",
            transform=ax.transAxes, va="top", fontsize=8.8, family="monospace")

    fig.suptitle("brAIn · M23 — O organismo vivo roda no substrato esparso: mesma cognição, menos energia",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m23_efficient_organism.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Comunicacao:  denso {comm_d[-1]:.0f}% -> esparso {comm_s[-1]:.0f}%")
    print(f"Conceitos:    denso {disc_d*100:.0f}% -> esparso {disc_s*100:.0f}%")
    print(f"Atividade:    denso {af_d*100:.1f}% -> esparso {af_s*100:.1f}%")
    print(f"SynOps/percep: denso {syn_d:.0f} -> esparso {syn_s:.0f}  (fator {factor:.1f}x)")


if __name__ == "__main__":
    main()
