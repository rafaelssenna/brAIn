"""Testes do M4 — garante que a matemática do predictive coding está correta.

Roda com:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import PredictiveCoder  # noqa: E402


def _prototypes(n=4, d=16, seed=1):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, size=(n, d))
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def test_inferencia_reduz_erro():
    """Após inferir r, o erro de previsão é menor do que com r=0."""
    pc = PredictiveCoder(n_obs=16, n_latent=4, seed=0)
    x = _prototypes()[0]
    err_zero = np.sum((x - pc.W @ np.zeros(4)) ** 2)   # erro com r=0
    err_inf = pc.prediction_error(x)                   # erro após inferência
    assert err_inf <= err_zero + 1e-9


def test_surpresa_cai_com_o_aprendizado():
    """A peça central do M4: treinando nos mesmos padrões, o erro DESPENCA."""
    P = _prototypes(n=4, d=16)
    pc = PredictiveCoder(n_obs=16, n_latent=6, eta_w=0.05, seed=0)

    err_inicial = np.mean([pc.prediction_error(p) for p in P])
    rng = np.random.default_rng(0)
    for _ in range(3000):
        x = P[rng.integers(len(P))]
        pc.learn(x)
    err_final = np.mean([pc.prediction_error(p) for p in P])

    assert err_final < 0.2 * err_inicial   # cai pelo menos 5x


def test_surpresa_noisy_aproxima_piso_de_subespaco():
    """Com ruído, o erro residual após aprender deve aproximar o ÓTIMO de um
    modelo de subespaço σ²(D−d) — não o zero (o ruído fora do subespaço é
    irredutível) — e ser medido em dados held-out (W congelado)."""
    sigma = 0.1
    P = _prototypes(n=4, d=16)               # 4 padrões em R^16
    rank = np.linalg.matrix_rank(P)          # dimensão do subespaço
    floor = sigma ** 2 * (16 - rank)         # melhor resíduo possível
    base_zero = 1.0 + sigma ** 2 * 16        # "não prever nada" (protótipos têm norma 1)

    pc = PredictiveCoder(n_obs=16, n_latent=8, eta_w=0.05, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        x = P[rng.integers(len(P))] + sigma * rng.standard_normal(16)
        pc.learn(x)

    # Held-out: W congelado, fluxo novo.
    held = np.mean([pc.prediction_error(P[rng.integers(len(P))] + sigma * rng.standard_normal(16))
                    for _ in range(2000)])
    assert held < 0.3 * base_zero            # aprendeu muito (longe de "prever nada")
    assert held < floor + 0.06               # chega perto do ótimo de subespaço
    assert held > floor - 0.02               # mas não abaixo do irredutível


def test_estrutura_emerge_nos_pesos():
    """Após aprender, algum campo receptivo (coluna de W) bate com cada protótipo."""
    P = _prototypes(n=3, d=16)
    pc = PredictiveCoder(n_obs=16, n_latent=6, eta_w=0.05, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(4000):
        pc.learn(P[rng.integers(len(P))])

    for p in P:
        # melhor alinhamento (|cosseno|) entre p e alguma coluna de W
        cos = np.abs(pc.W.T @ p)  # colunas já têm norma 1; p também
        assert cos.max() > 0.9


def test_learn_batch_tambem_reduz_surpresa():
    """A versão vetorizada (minibatch) também aprende: surpresa cai com o treino."""
    P = _prototypes(n=4, d=16)
    pc = PredictiveCoder(n_obs=16, n_latent=6, eta_w=0.05, seed=0)
    rng = np.random.default_rng(0)
    err_inicial = np.mean([pc.prediction_error(p) for p in P])
    for _ in range(400):
        batch = P[rng.integers(len(P), size=16)]      # lote de 16
        pc.learn_batch(batch + 0.05 * rng.standard_normal((16, 16)))
    err_final = np.mean([pc.prediction_error(p) for p in P])
    assert err_final < 0.3 * err_inicial


def test_pesos_permanecem_normalizados():
    """As colunas de W mantêm norma ~1 ao longo do aprendizado."""
    pc = PredictiveCoder(n_obs=16, n_latent=4, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(500):
        pc.learn(rng.normal(0, 1, 16))
    norms = np.linalg.norm(pc.W, axis=0)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_representacao_nao_negativa():
    """Com nonneg=True, a representação inferida nunca é negativa."""
    pc = PredictiveCoder(n_obs=16, n_latent=4, nonneg=True, seed=0)
    r = pc.infer(_prototypes()[0])
    assert np.all(r >= 0.0)
