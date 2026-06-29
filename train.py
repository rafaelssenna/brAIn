"""Treinador intensivo do brAIn — viver muito, consolidar (sono) e GUARDAR o aprendido.

Diferente de um treino de Transformer (uma vez, congelado), aqui o organismo VIVE: percebe
cenas, ouve as palavras certas (atenção conjunta, como uma criança), e CONSOLIDA o que
aprendeu por replay — o mesmo papel do SONO no cérebro (reativar experiências para fixá-las,
M8). E SALVA o cérebro em checkpoints: assim o aprendizado ACUMULA entre sessões — ele acorda
mais esperto, não renasce bebê.

O que treina: as 6 palavras de português ancoradas na percepção (topo/meio/base/esquerda/
centro/direita) — o mesmo mundo do chat.py, então o cérebro treinado aqui é o que o chat carrega.

Uso:
  python train.py                 # treino padrão (continua de cérebro salvo, se houver)
  python train.py --steps 200000  # nº de passos de vida
  python train.py --fresh         # começar do zero (ignora cérebro salvo)
  python train.py --out brain.npz # onde salvar o cérebro

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from brain import LivingAgent  # noqa: E402

np.seterr(all="ignore")

SIDE = 8
D = SIDE * SIDE
NOISE = 0.06
ROWS, COLS = [1, 4, 6], [1, 4, 6]
WORDS = ["topo", "meio", "base", "esquerda", "centro", "direita"]
K = len(WORDS)
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "brain.npz")


def objects():
    pats = []
    for r in ROWS:
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    for c in COLS:
        p = np.zeros((SIDE, SIDE)); p[:, c] = 1.0; pats.append(p.ravel())
    P = np.array(pats)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


P = objects()


def naming(agent):
    return float(np.mean([agent.speak(agent.concept(P[i]), explore=False) == i for i in range(K)]))


def comprehension(agent):
    cs = [agent.concept(P[j]) for j in range(K)]
    tot = 0.0
    for w in range(K):
        cands = [j for j in range(K) if cs[j] == agent.listen(w)]
        if w in cands:
            tot += 1.0 / len(cands)
    return tot / K


def discrim(agent):
    return agent.discriminability([P[i] for i in range(K)])


def live_step(agent, rng):
    """Um instante de vida: vê uma cena, percebe, e o 'professor' diz a palavra certa."""
    i = int(rng.integers(K))
    obj = P[i] + NOISE * rng.standard_normal(D)
    obj = obj / np.linalg.norm(obj)
    c = agent.concept(obj)
    agent.learn_perception(obj)          # percebe (com replay/sono embutido no LivingAgent)
    agent.learn_listen(i, c)             # ouve a palavra certa (atenção conjunta)
    agent.reinforce_speak(c, i, True)    # associa o conceito à palavra


def main():
    ap = argparse.ArgumentParser(description="Treinador intensivo do brAIn")
    ap.add_argument("--steps", type=int, default=100_000, help="passos de vida")
    ap.add_argument("--out", default=DEFAULT_OUT, help="arquivo do cérebro (.npz)")
    ap.add_argument("--fresh", action="store_true", help="começar do zero")
    ap.add_argument("--report-every", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # continua de um cérebro salvo, ou nasce novo
    if not args.fresh and os.path.exists(args.out):
        agent = LivingAgent.load(args.out)
        print(f"continuando o cérebro salvo em {args.out} "
              f"(já viveu {agent._t} passos; nomear atual {100*naming(agent):.0f}%)")
    else:
        agent = LivingAgent(D, n_latent=12, n_symbols=K, seed=args.seed)
        print(f"cérebro NOVO (nasce sem saber: nomear {100*naming(agent):.0f}%)")

    rng = np.random.default_rng(args.seed + agent._t)   # semente varia com a idade
    print(f"vivendo {args.steps} passos (consolidando por replay/sono)...\n")
    print(f"{'passo':>8} | {'nomear':>7} | {'apontar':>8} | {'percepção':>10}")
    print("-" * 42)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        live_step(agent, rng)
        if step % args.report_every == 0 or step == args.steps:
            print(f"{step:>8} | {100*naming(agent):>6.0f}% | {100*comprehension(agent):>7.0f}% | "
                  f"{100*discrim(agent):>9.0f}%")
            agent.save(args.out)                          # checkpoint: guarda o progresso
    dt = time.time() - t0

    print("-" * 42)
    print(f"\nFEITO em {dt:.0f}s. Cérebro salvo em {args.out}")
    print(f"Ele agora: nomeia {100*naming(agent):.0f}%, aponta {100*comprehension(agent):.0f}%, "
          f"distingue {100*discrim(agent):.0f}% — total de {agent._t} passos vividos.")
    print("\nO chat carrega esse cérebro: rode  python chat.py  e ele acorda esperto.")
    print("Exemplos do que ele diz agora:")
    for i in range(K):
        said = WORDS[agent.speak(agent.concept(P[i]), explore=False)]
        mark = "OK" if said == WORDS[i] else "x"
        print(f'   vê "{WORDS[i]:9s}" -> diz "{said}"  {mark}')


if __name__ == "__main__":
    main()
