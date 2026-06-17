"""Varredura de parâmetros do M3 (helper, não é artefato de marco).

Procura a config que maximiza a taxa de sucesso e minimiza a distância final,
mantendo a arquitetura fixa. Uso: python experiments/_m3_sweep.py
"""

from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from m3_embodiment import DEFAULT, evaluate  # noqa: E402

GRID = dict(
    prox_thresh=[0.88, 0.92],
    w_inh=[6.0, 8.0],
    motor_bias=[(1.8, 2.2)],
    v_k=[0.7, 0.9],
)
N_TRIALS = 15


def main():
    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"Testando {len(combos)} configurações ({N_TRIALS} episódios cada)...\n",
          flush=True)
    results = []
    for combo in combos:
        cfg = dict(DEFAULT)
        for k, v in zip(keys, combo):
            if k == "motor_bias":
                cfg["motor_bias_l"], cfg["motor_bias_r"] = v
            else:
                cfg[k] = v
        m = evaluate(cfg, n_trials=N_TRIALS)
        results.append((m["b_success"], -m["b_final"], combo, m))
        print(f"  prox={combo[0]:.2f} w_inh={combo[1]:.0f} bias={combo[2]} "
              f"v_k={combo[3]:.1f}  ->  sucesso={m['b_success']:.0f}% "
              f"final={m['b_final']:.1f} (rand {m['r_success']:.0f}%/{m['r_final']:.1f})",
              flush=True)

    results.sort(reverse=True)
    print("\n=== TOP 5 ===")
    for succ, neg_final, combo, m in results[:5]:
        print(f"sucesso={succ:.0f}% final={-neg_final:.1f} | "
              f"prox={combo[0]:.2f} w_inh={combo[1]:.0f} "
              f"bias_l={combo[2][0]} bias_r={combo[2][1]} v_k={combo[3]:.1f}")


if __name__ == "__main__":
    main()
