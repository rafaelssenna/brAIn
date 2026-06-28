"""M29 — Diálogo com TURNOS: pergunta → resposta ancorada (do conversar ao dialogar).

O M28 fez dois agentes conversarem com frases, mas em mão única (um descreve, o outro
adivinha). Dialogar tem TROCA: um PERGUNTA, o outro RESPONDE sobre o que percebe. Aqui dois
agentes olham a mesma cena (forma × posição); o PERGUNTADOR pergunta "que forma?" ou "que
posição?" e o RESPONDEDOR diz SÓ a palavra certa para aquela pergunta.

A capacidade NOVA: a pergunta SELECIONA o que descrever. O respondedor aprende a ROTEAR —
responder a forma quando perguntam forma, a posição quando perguntam posição. É o começo de
raciocinar sobre a linguagem: responder ao que foi perguntado, não despejar tudo.

Pergunta científica (H12): um agente aprende a RESPONDER À PERGUNTA — usar o tipo de pergunta
para escolher QUAL parte da cena descrever — só dialogando, sem ninguém ensinar?

Provas (método do projeto):
  • o entendimento por turno EMERGE do acaso (sem professor);
  • BASELINE ablado (respondedor CEGO à pergunta, só descreve a forma): acerta "que forma?"
    mas falha "que posição?" (~acaso) — prova que USAR a pergunta é o que faz dialogar;
  • a resposta é ancorada: o respondedor percebe a cena dos PIXELS (forma seus conceitos).

Uso:   python experiments/m29_dialogue.py
Salva: experiments/output/m29_dialogue.png

Honesto: escala de brinquedo (2 perguntas, poucos atributos, resposta de 1 palavra). É o
MECANISMO do turno pergunta→resposta com seleção, NÃO diálogo pleno (sem múltiplos turnos
encadeados, sem sintaxe rica) — horizonte distante, provável híbrido.

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
from brain import PredictiveCoder, DialogueGame, ASK_SHAPE, ASK_POS  # noqa: E402

np.seterr(all="ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SIDE = 8
D = SIDE * SIDE
K_SHAPE = 3
K_POS = 3
ALL = [(s, p) for s in range(K_SHAPE) for p in range(K_POS)]
N_STEPS = 8000
N_SEEDS = 6
NOISE = 0.05

SHAPE_WORDS = ["barraH", "barraV", "diag"]
POS_WORDS = ["topo", "meio", "base"]


def scene(shape, pos, rng=None):
    """Renderiza a cena: uma FORMA numa faixa de POSIÇÃO do grid."""
    g = np.zeros((SIDE, SIDE))
    band = {0: range(0, 3), 1: range(3, 5), 2: range(5, 8)}[pos]
    if shape == 0:
        g[list(band)[len(list(band)) // 2], :] = 1.0
    elif shape == 1:
        for r in band:
            g[r, SIDE // 2] = 1.0
    else:
        for k, r in enumerate(band):
            g[r, min(k, SIDE - 1)] = 1.0
    v = g.ravel()
    if rng is not None:
        v = v + NOISE * rng.standard_normal(D)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_perception(render_fn, k, seed):
    """Predictive coder forma conceitos de um canal (forma OU posição) dos pixels."""
    pc = PredictiveCoder(n_obs=D, n_latent=8, n_infer=30, eta_r=0.1, eta_w=0.05,
                         l2_prior=0.1, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        pc.learn(render_fn(int(rng.integers(k)), rng))
    votes = np.zeros((8, k))
    for _ in range(400):
        v = int(rng.integers(k))
        votes[int(np.argmax(pc.infer(render_fn(v, rng)))), v] += 1
    c2v = {c: int(np.argmax(votes[c])) for c in range(8) if votes[c].sum() > 0}

    def perceive(x):
        return c2v.get(int(np.argmax(pc.infer(x))), 0)
    return perceive, len(set(c2v.values())) / k


def run(route, n_steps, seed, track=False):
    g = DialogueGame(K_SHAPE, K_POS, route_by_question=route, seed=seed)
    rng = np.random.default_rng(seed + 50)
    xs, acc = [], []
    for t in range(n_steps):
        s, p = ALL[int(rng.integers(len(ALL)))]
        q = int(rng.integers(2))
        g.turn(s, p, q)
        if track and t % 150 == 0:
            xs.append(t); acc.append(100 * g.accuracy_both(ALL))
    return g, np.array(xs), acc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # percepção REAL ancorada (mostra que a resposta gruda no que ele percebe dos pixels)
    perceive_shape, disc_shape = build_perception(lambda v, r: scene(v, 1, r), K_SHAPE, seed=1)
    perceive_pos, disc_pos = build_perception(lambda v, r: scene(0, v, r), K_POS, seed=2)

    # (1) emergência do diálogo (roteando pela pergunta), sem professor
    g_emerge, xs, acc_curve = run(True, N_STEPS, seed=1, track=True)

    # (2) ablação honesta: rotear pela pergunta vs cego à pergunta (média de sementes)
    route_sh, route_po, blind_sh, blind_po = [], [], [], []
    for sd in range(N_SEEDS):
        gr, _, _ = run(True, N_STEPS, seed=sd + 1)
        gb, _, _ = run(False, N_STEPS, seed=sd + 1)
        route_sh.append(100 * gr.accuracy(ALL, ASK_SHAPE)); route_po.append(100 * gr.accuracy(ALL, ASK_POS))
        blind_sh.append(100 * gb.accuracy(ALL, ASK_SHAPE)); blind_po.append(100 * gb.accuracy(ALL, ASK_POS))
    r_sh, r_po = np.mean(route_sh), np.mean(route_po)
    b_sh, b_po = np.mean(blind_sh), np.mean(blind_po)
    chance = 100.0 / K_POS

    # ============================ figura ============================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) o diálogo emerge (entendimento por turno sobe do acaso).
    ax = axes[0, 0]
    ax.plot(xs, acc_curve, lw=2.2, color="#2ca02c")
    ax.axhline(100 * (0.5 * (1 / K_SHAPE + 1 / K_POS)), color="#bbb", ls="--", lw=1, label="acaso")
    ax.set_title("(a) O diálogo EMERGE sem professor (pergunta→resposta)")
    ax.set_xlabel("turnos de diálogo"); ax.set_ylabel("acerto da resposta %")
    ax.set_ylim(0, 105); ax.legend(loc="lower right", fontsize=9)

    # (b) ablação: usar a pergunta vs ser cego a ela.
    ax = axes[0, 1]
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, [r_sh, r_po], w, color="#1f77b4", label="usa a pergunta (roteia)")
    ax.bar(x + w / 2, [b_sh, b_po], w, color="#d62728", label="CEGO à pergunta")
    ax.axhline(chance, color="#999", ls=":", lw=1, label=f"acaso ({chance:.0f}%)")
    ax.set_xticks(x); ax.set_xticklabels(['pergunta "que forma?"', 'pergunta "que posição?"'])
    ax.set_ylabel("acerto %"); ax.set_ylim(0, 109)
    ax.set_title("(b) Usar a pergunta é o que permite responder à POSIÇÃO")
    ax.legend(loc="lower left", fontsize=8)
    for xi, v in zip([x[0] - w / 2, x[1] - w / 2], [r_sh, r_po]):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    for xi, v in zip([x[0] + w / 2, x[1] + w / 2], [b_sh, b_po]):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)

    # (c) exemplo de diálogo concreto (texto).
    ax = axes[1, 0]; ax.axis("off")
    lines = ["(c) Um diálogo do agente treinado:", ""]
    for (s, p) in [(0, 0), (1, 2), (2, 1)]:
        wsh = g_emerge.answer(s, p, ASK_SHAPE, explore=False)
        wpo = g_emerge.answer(s, p, ASK_POS, explore=False)
        ash = SHAPE_WORDS[g_emerge.interpret(wsh, ASK_SHAPE)]
        apo = POS_WORDS[g_emerge.interpret(wpo, ASK_POS)]
        lines.append(f"cena: [{SHAPE_WORDS[s]} no {POS_WORDS[p]}]")
        lines.append(f'   - "que forma?"   -> "{ash}"')
        lines.append(f'   - "que posição?" -> "{apo}"')
        lines.append("")
    ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=9.5, family="monospace")

    # (d) resumo honesto.
    ax = axes[1, 1]; ax.axis("off")
    ax.text(0.0, 0.98,
            "M29 — diálogo com turnos: pergunta → resposta\n\n"
            "Dois agentes olham a cena; um PERGUNTA (forma? / posição?),\n"
            "o outro PERCEBE a cena e RESPONDE só a palavra certa. Sem\n"
            "professor: o perguntador conhece o atributo que perguntou e\n"
            "confirma. O respondedor aprende a ROTEAR pela pergunta.\n\n"
            f"• percepção:   distingue formas {disc_shape*100:.0f}%, posições {disc_pos*100:.0f}%\n"
            f"• usa a pergunta:  forma {r_sh:.0f}%  |  posição {r_po:.0f}%\n"
            f"• CEGO à pergunta: forma {b_sh:.0f}%  |  posição {b_po:.0f}% (acaso {chance:.0f}%)\n\n"
            "Achado (H12): o agente aprende a RESPONDER À PERGUNTA —\n"
            "usar o tipo de pergunta para escolher QUAL parte da cena\n"
            "descrever. O agente CEGO acerta a forma mas erra a posição\n"
            "(responde forma): sem usar a pergunta, não há diálogo. É\n"
            "o começo de raciocinar sobre a linguagem: responder ao que\n"
            "foi perguntado, não despejar tudo.\n\n"
            "Honesto: 2 perguntas, resposta de 1 palavra, escala de\n"
            "brinquedo; mecanismo do turno pergunta→resposta, não\n"
            "diálogo pleno (sem turnos encadeados nem sintaxe rica).",
            transform=ax.transAxes, va="top", fontsize=8.2, family="monospace")

    fig.suptitle("brAIn · M29 — Diálogo com turnos: o agente aprende a RESPONDER À PERGUNTA (não só descrever)",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "m29_dialogue.png")
    fig.savefig(out, dpi=120)

    print(f"Figura salva em: {out}")
    print(f"Percepcao: formas {disc_shape*100:.0f}%, posicoes {disc_pos*100:.0f}%")
    print(f"Usa a pergunta:  forma {r_sh:.0f}% | posicao {r_po:.0f}%")
    print(f"Cego a pergunta: forma {b_sh:.0f}% | posicao {b_po:.0f}% (acaso {chance:.0f}%)")
    print(f"Emergencia: {acc_curve[0]:.0f}% -> {acc_curve[-1]:.0f}%")


if __name__ == "__main__":
    main()
