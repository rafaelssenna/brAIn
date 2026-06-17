"""M8 — Memória: vencer o esquecimento catastrófico (Fase 2).

Hipótese H4: com REPLAY (reensaio de experiências, como o sono), o agente retém
o que aprendeu mesmo ao aprender coisas novas — e isso pode destravar a vantagem
da curiosidade (que nos marcos M5/M7 perdia pro aleatório só por esquecer).

Dois testes:
  (a/b) ESQUECIMENTO PURO: treina só o conjunto A, depois SÓ o B. Mede se A é
        lembrado. Sem replay: A decai. Com replay: A é retido.
  (c)   DESTRAVA A CURIOSIDADE? O organismo integrado (M7) com replay vs sem vs
        aleatório — o replay fecha (ou inverte) a diferença de domínio?

Uso:   python experiments/m8_memory.py
Salva: experiments/output/m8_memory.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import PredictiveCoder, ReplayBuffer, RingWorld, IntegratedAgent  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GRID = 8
D = GRID * GRID
NOISE = 0.05


def bars(idxs):
    pats = []
    for i in idxs:
        p = np.zeros((GRID, GRID))
        if i < GRID:
            p[i, :] = 1.0
        else:
            p[:, i - GRID] = 1.0
        pats.append(p.ravel())
    P = np.array(pats)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


A = bars([0, 2])             # conjunto A (aprendido primeiro)
B = bars([9, 11, 13, 15])    # conjunto B (aprendido depois, disjunto)


def err_on(pc, P):
    return float(np.mean([pc.prediction_error(p) for p in P]))


def sequential(use_replay, T1=1500, T2=2500, seed=SEED):
    """Fase A: só A. Fase B: só B (+ replay opcional). Mede o erro em A o tempo todo."""
    pc = PredictiveCoder(n_obs=D, n_latent=12, eta_w=0.05, l2_prior=0.1, seed=0)
    buf = ReplayBuffer(400, D, seed=seed) if use_replay else None
    rng = np.random.default_rng(seed)
    xs, errA, errB = [], [], []
    for t in range(T1 + T2):
        src = A if t < T1 else B
        x = src[rng.integers(len(src))] + NOISE * rng.standard_normal(D)
        x = x / np.linalg.norm(x)
        pc.learn(x)
        if buf is not None:
            buf.add(x)
            if t % 5 == 0 and buf.size:                 # reensaio periódico
                for xr in buf.sample(8):
                    pc.learn(xr)
        if t % 50 == 0:
            xs.append(t); errA.append(err_on(pc, A)); errB.append(err_on(pc, B))
    return np.array(xs), np.array(errA), np.array(errB), T1


def integrated(policy, replay, seed=SEED, n_steps=5000):
    """Roda o organismo do M7, opcionalmente com replay; devolve domínio final e %ruído."""
    world = RingWorld(n_locations=8, noise_locs=(3, 6), seed=1)
    kw = dict(replay_capacity=500, replay_every=10, replay_batch=8) if replay else {}
    agent = IntegratedAgent(world, policy=policy, seed=seed, **kw)
    rng = np.random.default_rng(seed)
    learn = world.learnable_locs()
    err0 = {l: agent.pc.prediction_error(world.patterns[l]) for l in learn}
    noise = 0
    for t in range(n_steps):
        here, _ = agent.step(rng)
        noise += int(here in (3, 6))
    m = agent.mastery()
    mastery = 100 * np.mean([max(0.0, 1 - m[l] / (err0[l] + 1e-9)) for l in learn])
    return mastery, 100 * noise / n_steps


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    xs, errA_no, errB_no, T1 = sequential(use_replay=False)
    _, errA_yes, errB_yes, _ = sequential(use_replay=True)
    forget_no = errA_no[-1] - errA_no[xs < T1][-1]
    forget_yes = errA_yes[-1] - errA_yes[xs < T1][-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

    # (a) erro em A ao longo das fases, com vs sem replay.
    ax = axes[0, 0]
    ax.plot(xs, errA_no, lw=2, color="#d62728", label="A — SEM replay (esquece)")
    ax.plot(xs, errA_yes, lw=2, color="#2ca02c", label="A — COM replay (lembra)")
    ax.axvline(T1, ls="--", lw=1, color="k")
    ax.text(T1 * 0.5, ax.get_ylim()[1] * 0.9, "fase A:\nsó A", ha="center", fontsize=8)
    ax.text(T1 + (xs[-1] - T1) * 0.5, ax.get_ylim()[1] * 0.9, "fase B:\nsó B", ha="center", fontsize=8)
    ax.set_title("(a) O fácil é lembrado mesmo aprendendo o novo?")
    ax.set_xlabel("passo"); ax.set_ylabel("erro de previsão em A (held-out)")
    ax.legend(loc="center right", fontsize=8.5)

    # (b) esquecimento (Δ erro em A na fase B): sem vs com.
    ax = axes[0, 1]
    bars_ = ax.bar(["sem replay", "com replay"], [forget_no, forget_yes],
                   color=["#d62728", "#2ca02c"])
    for b, v in zip(bars_, [forget_no, forget_yes]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("(b) Esquecimento do A na fase B (menor = melhor)")
    ax.set_ylabel("Δ erro em A")

    # (c) o replay destrava a curiosidade no organismo integrado?
    ax = axes[1, 0]
    m_cur_replay, n_cur_replay = integrated("curious", replay=True)
    m_cur, n_cur = integrated("curious", replay=False)
    m_ran, n_ran = integrated("random", replay=False)
    labels = ["curioso\n+ replay", "curioso\n(M7)", "aleatório"]
    vals = [m_cur_replay, m_cur, m_ran]
    cols = ["#2ca02c", "#9467bd", "#888888"]
    bb = ax.bar(labels, vals, color=cols)
    for b, v in zip(bb, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=10)
    ax.set_title("(c) Domínio final no organismo integrado")
    ax.set_ylabel("domínio dos lugares (%)"); ax.set_ylim(0, 105)

    # (d) resumo.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M8 — memória por REPLAY (como o sono)\n\n"
            f"Esquecimento do A na fase B:\n"
            f"  • sem replay: {forget_no:+.3f}  (esquece)\n"
            f"  • com replay: {forget_yes:+.3f}  (retém)\n\n"
            f"Organismo integrado (domínio / %ruído):\n"
            f"  • curioso + replay: {m_cur_replay:.0f}% / {n_cur_replay:.0f}%\n"
            f"  • curioso (M7):     {m_cur:.0f}% / {n_cur:.0f}%\n"
            f"  • aleatório:        {m_ran:.0f}% / {n_ran:.0f}%\n\n"
            "Mecanismo biológico: reensaio de experiências guardadas\n"
            "(replay hipocampal). Amostragem por reservatório mantém\n"
            "memórias de TODA a vida, não só as recentes.",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle("brAIn · M8 — Memória: replay vence o esquecimento catastrófico",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m8_memory.png")
    fig.savefig(out, dpi=120)
    print(f"Figura salva em: {out}")
    print(f"Esquecimento do A  sem replay={forget_no:+.3f}  com replay={forget_yes:+.3f}")
    print(f"Integrado domínio  curioso+replay={m_cur_replay:.0f}%  curioso={m_cur:.0f}%  aleatório={m_ran:.0f}%")
    print(f"Integrado %ruído   curioso+replay={n_cur_replay:.0f}%  curioso={n_cur:.0f}%  aleatório={n_ran:.0f}%")


if __name__ == "__main__":
    main()
