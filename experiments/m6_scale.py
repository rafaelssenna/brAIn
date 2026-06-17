"""M6 — Escala: vetorização e o gargalo honesto.

O laço em Python puro (uma amostra por vez) é o gargalo do projeto. Aqui medimos
o ganho de VETORIZAR (processar um lote de amostras de uma vez, amortizando o
custo fixo do NumPy) e rodamos um mundo MAIOR para mostrar que a matemática
escala — sendo honesto sobre onde o Python puro ainda limita.

Uso:   python experiments/m6_scale.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import PredictiveCoder  # noqa: E402

SEED = 42


def make_bars(grid, n_bars):
    """n_bars barras (linhas/colunas) num grid, norma 1."""
    pats, i = [], 0
    while len(pats) < n_bars:
        p = np.zeros((grid, grid))
        if i % 2 == 0:
            p[(i // 2) % grid, :] = 1.0
        else:
            p[:, (i // 2) % grid] = 1.0
        pats.append(p.ravel()); i += 1
    P = np.array(pats)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def _surprise(pc, P):
    return float(np.mean([pc.prediction_error(p) for p in P]))


def bench(grid, n_patterns, n_samples, batch, seed=SEED):
    """Compara JUSTO: quanto tempo de relógio até atingir a MESMA qualidade.

    - online: 1 passada de n_samples amostras (uma por vez).
    - lote: épocas vetorizadas até atingir a surpresa do online (cada época é
      barata; um update de lote substitui B updates online, então precisa de mais
      épocas — mas elas são ~20x mais rápidas).
    """
    D = grid * grid
    P = make_bars(grid, n_patterns)
    rng = np.random.default_rng(seed)
    stream = P[rng.integers(n_patterns, size=n_samples)] + 0.06 * rng.standard_normal((n_samples, D))
    nlat = n_patterns + 4

    pc1 = PredictiveCoder(n_obs=D, n_latent=nlat, eta_w=0.05, seed=0)
    t0 = time.perf_counter()
    for x in stream:
        pc1.learn(x)
    t_online = time.perf_counter() - t0
    err_online = _surprise(pc1, P)
    thr_online = n_samples / t_online

    pc2 = PredictiveCoder(n_obs=D, n_latent=nlat, eta_w=0.05, seed=0)
    t0, epochs, total = time.perf_counter(), 0, 0
    while epochs < 80:
        for i in range(0, n_samples, batch):
            pc2.learn_batch(stream[i:i + batch]); total += stream[i:i + batch].shape[0]
        epochs += 1
        if _surprise(pc2, P) <= err_online:        # alcançou a qualidade do online
            break
    t_batch = time.perf_counter() - t0
    thr_batch = total / t_batch
    return dict(t_online=t_online, err_online=err_online, thr_online=thr_online,
                t_batch=t_batch, err_batch=_surprise(pc2, P), epochs=epochs,
                thr_batch=thr_batch)


def main():
    print("=== Escala: online (1 a 1) vs lote vetorizado (mesma qualidade-alvo) ===")
    for grid, npat, ns, b in [(8, 8, 6000, 64), (12, 18, 6000, 64)]:
        r = bench(grid, npat, ns, b)
        print(f"grid {grid}x{grid} ({npat} padrões, {ns} amostras):")
        print(f"   online: {r['t_online']:5.1f}s p/ surpresa {r['err_online']:.3f}  "
              f"({r['thr_online']:,.0f} amostras/s)")
        print(f"   lote  : {r['t_batch']:5.1f}s p/ surpresa {r['err_batch']:.3f} "
              f"em {r['epochs']} épocas  ({r['thr_batch']:,.0f} amostras/s)")
        ratio = r["t_batch"] / r["t_online"]
        verdict = (f"{ratio:.1f}x MAIS LENTO no relógio total" if ratio > 1
                   else f"{1 / ratio:.1f}x mais rápido")
        print(f"   -> throughput {r['thr_batch'] / r['thr_online']:.0f}x, MAS precisou de "
              f"{r['epochs']} épocas p/ igualar => no total: {verdict}")
    print("\nHonesto (resultado negativo): aqui o gargalo é EFICIÊNCIA AMOSTRAL")
    print("(nº de updates), não throughput. Um update de lote substitui B online, então")
    print("vetorizar ingenuamente NÃO vence em escala de brinquedo. O ganho real da")
    print("vetorização aparece em modelos GRANDES (ops matriciais dominam) em GPU, ou")
    print("com hardware neuromórfico para o online spiking. NumPy/CPU não é o caminho.")


if __name__ == "__main__":
    main()
