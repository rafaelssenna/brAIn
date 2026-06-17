"""Predictive coding — M4 (a máquina de previsão).

Implementação do zero (NumPy) de uma rede de codificação preditiva à
Rao & Ballard (1999). Ver docs/04-fundacoes-matematicas.md, seção 3.

Ideia central: o cérebro é uma máquina de PREVISÃO. Uma camada de causas
latentes `r` tenta PREVER a entrada sensorial `x`; o que não é previsto vira
ERRO `ε = x - x̂`, e a rede aprende minimizando esse erro.

  Previsão:   x̂ = W r
  Erro:       ε = x - x̂
  Energia:    E = ½‖ε‖²  (+ prior L2/Gaussiano ½λ‖r‖²)

Dois processos em escalas de tempo diferentes:

  1) INFERÊNCIA (rápida, dentro de uma observação): ajusta `r` para reduzir a
     energia, descendo o gradiente:
        dr/dt ∝ -∂E/∂r = Wᵀ ε - λ r
  2) APRENDIZADO (lento, ao longo da vida): ajusta `W` com uma regra LOCAL:
        ΔW ∝ ε rᵀ
     — usa só o erro na saída da conexão e a representação na entrada. Nenhum
     gradiente global, NENHUM backpropagation. É a candidata biologicamente
     plausível ao "aprendizado do córtex".

Sem dataset, sem pré-treino: a rede nasce com W aleatório e aprende a estrutura
do seu mundo vivendo-o.

ESCOPO E HONESTIDADE (lacunas reais, ver docs/05 e a verificação adversarial):
  • Esta é a instância LINEAR, de UMA camada, precisão = I, da forma geral
    (não-linear, hierárquica, com precisão Π) descrita em docs/04 seção 3.
  • Numa única camada, a regra local ΔW ∝ ε rᵀ É exatamente o gradiente -∂E/∂W.
    Logo "local" e "gradiente" coincidem aqui — o ganho do predictive coding
    sobre o backprop (evitar o "weight transport") só aparece numa HIERARQUIA,
    ainda não implementada. A afirmação "sem backprop" é literal e verdadeira,
    mas o mérito é modesto enquanto for uma camada só.
  • Módulo RATE-BASED (contínuo): NÃO usa spiking nem STDP. M1–M4 validam cada
    ingrediente ISOLADAMENTE; o sistema integrado (spiking+STDP+predição,
    corporificado) que a H1 descreve ainda NÃO existe.
"""

from __future__ import annotations

import numpy as np


class PredictiveCoder:
    """Uma camada de codificação preditiva (entrada D -> latente N)."""

    def __init__(
        self,
        n_obs: int,            # dimensão da entrada sensorial (D)
        n_latent: int,         # nº de causas latentes / unidades de representação (N)
        n_infer: int = 30,     # iterações de inferência por observação
        eta_r: float = 0.1,    # taxa da inferência (rápida)
        eta_w: float = 0.01,   # taxa do aprendizado (lenta)
        l2_prior: float = 0.1, # prior L2/Gaussiano sobre r (encolhe r; NÃO é esparsidade L1)
        nonneg: bool = True,   # representações não-negativas (faz a maior parte do "esparsificar")
        seed: int = 0,
    ) -> None:
        self.n_obs = n_obs
        self.n_latent = n_latent
        self.n_infer = n_infer
        self.eta_r = eta_r
        self.eta_w = eta_w
        self.l2_prior = l2_prior
        self.nonneg = nonneg

        rng = np.random.default_rng(seed)
        # W: cada COLUNA é um "campo receptivo"/template que a rede pode prever.
        self.W = rng.normal(0.0, 0.1, size=(n_obs, n_latent))
        self._normalize_columns()

    def _normalize_columns(self) -> None:
        """Mantém cada coluna de W com norma 1 (estabilidade do aprendizado)."""
        norms = np.linalg.norm(self.W, axis=0, keepdims=True)
        self.W = self.W / (norms + 1e-8)

    def predict(self, r: np.ndarray) -> np.ndarray:
        """Previsão descendente x̂ = W r."""
        return self.W @ r

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Inferência: ajusta r para minimizar a energia (gradiente local)."""
        r = np.zeros(self.n_latent)
        for _ in range(self.n_infer):
            err = x - self.W @ r                       # ε = x - x̂
            dr = self.W.T @ err - self.l2_prior * r    # -∂E/∂r
            r = r + self.eta_r * dr
            if self.nonneg:
                r = np.maximum(r, 0.0)
        return r

    def learn(self, x: np.ndarray):
        """Uma "experiência": infere r e aplica a regra LOCAL ΔW ∝ ε rᵀ.

        Retorna (r, erro_de_previsão) onde o erro é o ‖ε‖² após a inferência.
        """
        r = self.infer(x)
        err = x - self.W @ r
        self.W += self.eta_w * np.outer(err, r)   # regra de Hebb sobre o erro
        self._normalize_columns()
        return r, float(np.sum(err ** 2))

    def prediction_error(self, x: np.ndarray) -> float:
        """Erro de previsão (‖ε‖²) para uma entrada, SEM aprender."""
        r = self.infer(x)
        err = x - self.W @ r
        return float(np.sum(err ** 2))

    def learn_batch(self, X: np.ndarray) -> float:
        """Versão VETORIZADA (minibatch) de learn() — M6 (escala).

        Processa B amostras de uma vez (X de shape (B, D)) com a mesma matemática,
        amortizando o custo fixo do NumPy por chamada. Minibatch ≈ média do
        gradiente online (não é o online puro, mas é o caminho para escalar).
        Retorna a surpresa média ‖ε‖² do lote após a inferência.
        """
        X = np.atleast_2d(X)
        B = X.shape[0]
        R = np.zeros((B, self.n_latent))
        for _ in range(self.n_infer):                 # inferência vetorizada no lote
            err = X - R @ self.W.T                     # (B, D)
            dr = err @ self.W - self.l2_prior * R      # (B, N)
            R = R + self.eta_r * dr
            if self.nonneg:
                R = np.maximum(R, 0.0)
        err = X - R @ self.W.T
        self.W += self.eta_w * (err.T @ R) / B         # gradiente médio do lote
        self._normalize_columns()
        return float(np.mean(np.sum(err ** 2, axis=1)))
