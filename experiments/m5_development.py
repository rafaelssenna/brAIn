"""M5 — Desenvolvimento: curiosidade, estágios emergentes, e não esquecer.

Testa a hipótese H2: a aquisição de competência segue ESTÁGIOS que emergem
SOZINHOS, sem serem programados — guiados por motivação intrínseca (curiosidade
= learning progress, à la Oudeyer).

O mundo tem 4 "atividades", de dificuldades diferentes:
  • fácil   — 1 padrão
  • médio   — 3 padrões
  • difícil — 6 padrões
  • RUÍDO   — chiado puro, inaprendível (a "TV chiando")

O agente (um predictive coder do M4) escolhe a cada instante ONDE prestar
atenção. Demonstra:

  (a) A ORDEM fácil→médio→difícil vem da DIFICULDADE intrínseca (o aleatório faz
      igual) — NÃO é a curiosidade que a cria. O ganho da política está em (b).
  (b) CURRÍCULO próprio (este SIM é efeito da política): a atenção migra
      fácil→médio→difícil e ABANDONA o ruído.
  (c) Curiosidade BATE a novelty-seeking (que afunda na "TV chiando"). Honestamente
      NÃO supera o aleatório nesta tarefa fácil.
  (d) CONTINUAL LEARNING: ao aprender o difícil, esquece o que já sabia?

Uso:   python experiments/m5_development.py
Salva: experiments/output/m5_development.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import PredictiveCoder, IntrinsicMotivation, select_region  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GRID = 8
D = GRID * GRID
NOISE_TRAIN = 0.05
# 3 regiões aprendíveis + 3 de ruído (o mundo real é quase todo inaprendível:
# a alocação de atenção precisa importar para a comparação ser justa).
REGION_NAMES = ["fácil", "médio", "difícil", "ruído A", "ruído B", "ruído C"]
REGION_COLORS = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728", "#e377c2", "#ff7f0e"]
LEARNABLE = [0, 1, 2]
NOISE_REGIONS = [3, 4, 5]
N_REGIONS = 6


def make_pattern_pool():
    """10 padrões distintos (barras h/v/diagonais) num grid 8x8, norma 1."""
    pats = []
    for r in range(4):
        p = np.zeros((GRID, GRID)); p[r * 2, :] = 1.0; pats.append(p)     # 4 horizontais
    for c in range(4):
        p = np.zeros((GRID, GRID)); p[:, c * 2] = 1.0; pats.append(p)     # 4 verticais
    d1 = np.eye(GRID); d2 = np.fliplr(np.eye(GRID))
    pats.append(d1); pats.append(d2)                                      # 2 diagonais
    P = np.array([p.ravel() for p in pats])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


POOL = make_pattern_pool()
REGION_PATTERNS = {0: [0], 1: [1, 2, 3], 2: [4, 5, 6, 7, 8, 9],
                   3: None, 4: None, 5: None}   # 3,4,5 = ruído (inaprendível)


def sample(region, rng):
    """Observação da região: padrão + ruído (aprendível) ou chiado puro (ruído)."""
    if REGION_PATTERNS[region] is None:                 # TV chiando
        x = rng.standard_normal(D)
        return x / np.linalg.norm(x)                    # norma 1, mas sempre novo
    idx = REGION_PATTERNS[region][rng.integers(len(REGION_PATTERNS[region]))]
    x = POOL[idx] + NOISE_TRAIN * rng.standard_normal(D)
    return x / np.linalg.norm(x)   # mesma norma do held-out e do ruído (sem viés)


def heldout_error(pc, region):
    """Erro médio de previsão nos padrões LIMPOS da região (mede domínio)."""
    return float(np.mean([pc.prediction_error(POOL[i]) for i in REGION_PATTERNS[region]]))


def run_agent(policy, n_steps, seed=0, eval_every=100, warmup_per_region=15):
    """Roda um agente com a política dada. Registra atenção e domínio."""
    pc = PredictiveCoder(n_obs=D, n_latent=20, n_infer=30,
                         eta_r=0.1, eta_w=0.05, l2_prior=0.1, seed=seed)
    mot = IntrinsicMotivation(n_regions=N_REGIONS)
    rng = np.random.default_rng(seed)

    attention = np.zeros((n_steps, N_REGIONS))   # one-hot da região escolhida por passo
    checkpoints, mastery_curves = [], {k: [] for k in LEARNABLE}
    err0 = {k: heldout_error(pc, k) for k in LEARNABLE}   # erro inicial (baseline)

    warmup = warmup_per_region * N_REGIONS
    for t in range(n_steps):
        if t < warmup:
            region = t % N_REGIONS          # warmup: visita todas em rodízio
        else:
            region = select_region(mot, policy, rng)
        x = sample(region, rng)
        _, err = pc.learn(x)
        mot.update(region, err)
        attention[t, region] = 1.0

        if t % eval_every == 0:
            checkpoints.append(t)
            for k in LEARNABLE:
                # domínio (%): quanto do erro inicial já foi eliminado
                m = max(0.0, 1.0 - heldout_error(pc, k) / (err0[k] + 1e-9))
                mastery_curves[k].append(100.0 * m)

    return dict(pc=pc, attention=attention, checkpoints=np.array(checkpoints),
                mastery=mastery_curves)


def continual_learning_test(seed=0):
    """Treina só o fácil, depois SÓ o difícil; mede se esquece o fácil."""
    pc = PredictiveCoder(n_obs=D, n_latent=20, n_infer=30,
                         eta_r=0.1, eta_w=0.05, l2_prior=0.1, seed=seed)
    rng = np.random.default_rng(seed)
    T1, T2 = 1500, 2500
    easy_curve, hard_curve, xs = [], [], []
    for t in range(T1 + T2):
        region = 0 if t < T1 else 2          # fase A: fácil; fase B: SÓ difícil
        pc.learn(sample(region, rng))
        if t % 50 == 0:
            xs.append(t)
            easy_curve.append(heldout_error(pc, 0))   # o fácil ainda é lembrado?
            hard_curve.append(heldout_error(pc, 2))
    return np.array(xs), np.array(easy_curve), np.array(hard_curve), T1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_steps = 4500

    cur = run_agent("curiosity", n_steps, seed=SEED)
    nov = run_agent("novelty", n_steps, seed=SEED)
    ran = run_agent("random", n_steps, seed=SEED)

    def total_mastery(run):
        return np.mean([run["mastery"][k] for k in LEARNABLE], axis=0)

    WARMUP = 15 * N_REGIONS   # passos de rodízio inicial (não são da política)

    def noise_time(run):
        # mede SÓ pós-warmup (o rodízio inicial não reflete a política)
        return 100.0 * run["attention"][WARMUP:, NOISE_REGIONS].sum(axis=1).mean()

    def steps_to(run, thresh=80.0):
        """Quantos passos até o domínio total cruzar `thresh`% (∞ se nunca)."""
        tm = total_mastery(run)
        hit = np.flatnonzero(tm >= thresh)
        return int(run["checkpoints"][hit[0]]) if len(hit) else None

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (a) ESTÁGIOS: domínio por região ao longo da vida (agente curioso).
    ax = axes[0, 0]
    for k in LEARNABLE:
        ax.plot(cur["checkpoints"], cur["mastery"][k], lw=2,
                color=REGION_COLORS[k], label=REGION_NAMES[k])
        ax.plot(ran["checkpoints"], ran["mastery"][k], lw=1.2, ls="--",
                color=REGION_COLORS[k], alpha=0.55)
    ax.plot([], [], color="k", lw=2, label="curiosidade")
    ax.plot([], [], color="k", lw=1.2, ls="--", label="aleatório")
    ax.set_title("(a) A ordem vem da DIFICULDADE (o aleatório, tracejado, faz igual)")
    ax.set_xlabel("experiência (passo)"); ax.set_ylabel("domínio (%)")
    ax.legend(loc="lower right", fontsize=7.5, ncol=2); ax.set_ylim(-3, 103)

    # (b) CURRÍCULO próprio: fração de atenção por região (janela deslizante).
    ax = axes[0, 1]
    win = 200
    t = np.arange(win - 1, n_steps)
    for r in LEARNABLE:
        frac = np.convolve(cur["attention"][:, r], np.ones(win) / win, mode="valid")
        ax.plot(t, 100 * frac, lw=1.8, color=REGION_COLORS[r], label=REGION_NAMES[r])
    # ruído agregado (as 3 regiões inaprendíveis juntas)
    noise_attn = cur["attention"][:, NOISE_REGIONS].sum(axis=1)
    frac = np.convolve(noise_attn, np.ones(win) / win, mode="valid")
    ax.plot(t, 100 * frac, lw=1.8, color="#d62728", ls=":",
            label="ruído (3 regiões)")
    ax.axvspan(0, 15 * N_REGIONS, color="gray", alpha=0.13)
    ax.text(15 * N_REGIONS + 20, 92, "warmup", fontsize=7, color="gray")
    ax.set_title("(b) Currículo próprio (efeito real da política): atenção migra")
    ax.set_xlabel("experiência (passo)"); ax.set_ylabel("atenção (% janela)")
    ax.legend(loc="upper right", fontsize=9)

    # (c) Curiosidade vs novelty vs aleatório.
    ax = axes[1, 0]
    ax.plot(cur["checkpoints"], total_mastery(cur), lw=2.2, color="#2ca02c",
            label=f"curiosidade (ruído: {noise_time(cur):.0f}% do tempo)")
    ax.plot(nov["checkpoints"], total_mastery(nov), lw=2.0, color="#ff7f0e",
            label=f"novelty-seeking (ruído: {noise_time(nov):.0f}%)")
    ax.plot(ran["checkpoints"], total_mastery(ran), lw=2.0, color="#888888", ls="--",
            label=f"aleatório (ruído: {noise_time(ran):.0f}%)")
    ax.set_title("(c) O sinal certo importa: novelty colapsa na 'TV chiando'")
    ax.set_xlabel("experiência (passo)")
    ax.set_ylabel("domínio total das regiões aprendíveis (%)")
    ax.legend(loc="center right", fontsize=8.5); ax.set_ylim(-3, 103)
    fin_cur, fin_ran = total_mastery(cur)[-1], total_mastery(ran)[-1]
    s_cur, s_ran = steps_to(cur), steps_to(ran)
    ax.text(0.03, 0.06,
            "honesto: aqui curiosidade NÃO supera o aleatório\n"
            f"(final {fin_cur:.0f}% vs {fin_ran:.0f}%; 80% em {s_cur} vs {s_ran}).\n"
            "ganho real = currículo ativo (b) + fugir do ruído.\n"
            "(parte do tempo no ruído é só o piso do eps-greedy)",
            transform=ax.transAxes, fontsize=7.5, style="italic")

    # (d) CONTINUAL LEARNING: esquece o fácil ao aprender o difícil?
    ax = axes[1, 1]
    xs, easy, hard, T1 = continual_learning_test(seed=SEED)
    ax.plot(xs, easy, lw=2, color="#2ca02c", label="erro no FÁCIL (já aprendido)")
    ax.plot(xs, hard, lw=2, color="#9467bd", label="erro no DIFÍCIL (novo)")
    ax.axvline(T1, ls="--", lw=1, color="k")
    ax.text(T1 * 0.5, ax.get_ylim()[1] * 0.9, "fase A:\nsó fácil", ha="center", fontsize=8)
    ax.text(T1 + (xs[-1] - T1) * 0.5, ax.get_ylim()[1] * 0.9, "fase B:\nsó difícil",
            ha="center", fontsize=8)
    forget = easy[-1] - easy[xs < T1][-1]   # baseline = último checkpoint puro da fase A
    ax.set_title(f"(d) Continual learning: esquecimento do fácil = {forget:+.3f}")
    ax.set_xlabel("passo"); ax.set_ylabel("erro de previsão (held-out)")
    ax.legend(loc="center right", fontsize=8.5)

    fig.suptitle("brAIn · M5 — Desenvolvimento: currículo ativo e fuga do ruído (a ordem dos estágios vem da dificuldade)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = os.path.join(OUT_DIR, "m5_development.png")
    fig.savefig(out_path, dpi=120)
    print(f"Figura salva em: {out_path}")
    print(f"Domínio total final  curiosidade={total_mastery(cur)[-1]:.0f}%  "
          f"novelty={total_mastery(nov)[-1]:.0f}%  aleatório={total_mastery(ran)[-1]:.0f}%")
    print(f"Tempo na 'TV chiando'  curiosidade={noise_time(cur):.0f}%  "
          f"novelty={noise_time(nov):.0f}%  aleatório={noise_time(ran):.0f}%")
    print(f"Passos até 80% de domínio  curiosidade={steps_to(cur)}  "
          f"novelty={steps_to(nov)}  aleatório={steps_to(ran)}")
    print(f"Esquecimento do fácil (fase B): {forget:+.3f} de erro")


if __name__ == "__main__":
    main()
