"""Treino PESADO no mundo rico — onde treinar muito finalmente rende.

O mundo pequeno saturava em ~20k passos. O mundo rico (192 objetos: forma×cor×tamanho×
posição, imagem colorida 16×16×3) NÃO satura rápido — há muito a perceber, distinguir e
nomear. Aqui o cérebro rico (`RichBrain`: um módulo cortical por atributo) vive milhões de
passos, consolida por replay (sono) e SALVA checkpoints. O aprendizado acumula entre sessões.

Uso:
  python train_rich.py                    # treino padrão (continua de cérebro salvo)
  python train_rich.py --steps 3000000    # treino pesado de verdade
  python train_rich.py --fresh            # do zero
  python train_rich.py --latent 48        # mais conceitos por módulo (mais capacidade)

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from brain import RichBrain  # noqa: E402
from brain.rich_world import all_objects, ATTRS, n_objects  # noqa: E402

np.seterr(all="ignore")

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "brain_rich")
OBJS = all_objects()


def report(brain, test):
    acc = brain.naming_accuracy(test)
    return acc, float(np.mean(list(acc.values())))


def main():
    ap = argparse.ArgumentParser(description="Treino pesado do brAIn no mundo rico")
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--report-every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.fresh and os.path.isdir(args.out):
        brain = RichBrain.load(args.out)
        print(f"continuando o cérebro rico em {args.out} (já viveu {brain.lived} passos)")
    else:
        brain = RichBrain(n_latent=args.latent, seed=args.seed)
        print(f"cérebro rico NOVO: {n_objects()} objetos possíveis, {brain.n_latent} conceitos/módulo")

    rng = np.random.default_rng(args.seed + brain.lived)
    test = [OBJS[i] for i in np.random.default_rng(123).choice(len(OBJS), 80, replace=False)]

    print(f"\nvivendo {args.steps} passos no mundo rico (consolidando por replay/sono)...\n")
    hdr = "    passo | " + " | ".join(f"{a[:5]:>6}" for a in ATTRS) + " |  média"
    print(hdr); print("-" * len(hdr))
    t0 = time.time()
    for step in range(1, args.steps + 1):
        brain.learn(OBJS[int(rng.integers(len(OBJS)))], rng)
        if step % args.report_every == 0 or step == args.steps:
            acc, mean = report(brain, test)
            row = f"{step:>9} | " + " | ".join(f"{100*acc[a]:>5.0f}%" for a in ATTRS) + f" | {100*mean:>5.0f}%"
            print(row)
            brain.save(args.out)
    dt = time.time() - t0

    acc, mean = report(brain, test)
    print("-" * len(hdr))
    print(f"\nFEITO em {dt/60:.1f} min. Cérebro rico salvo em {args.out}/  ({brain.lived} passos vividos).")
    print(f"Nomear (média dos 4 atributos): {100*mean:.0f}%")
    print("\nExemplos do que ele descreve agora:")
    from brain.rich_world import VOCAB
    for o in test[:5]:
        said = brain.describe(o)
        true = " ".join(_attr_word(a, o[a]) for a in range(4))
        got = " ".join(VOCAB[w] for w in said)
        mark = "OK" if got == true else "~"
        print(f'   objeto [{true}] -> diz "{got}"  {mark}')


def _attr_word(a, v):
    from brain.rich_world import ATTR_VALUES
    return ATTR_VALUES[a][v]


if __name__ == "__main__":
    main()
