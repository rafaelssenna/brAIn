"""Predictive coding ESPARSO e dirigido a eventos — M22 (método > força bruta).

O cérebro roda com <20W. A eficiência dele não vem de força bruta: vem de MÉTODO —
poucos neurônios ativos por vez (código esparso, ~5-10% como no córtex) e computação
DIRIGIDA A EVENTOS (só quem dispara custa). Este módulo encarna esse princípio sobre
a máquina de previsão do M4, SEM abandonar a regra local (sem backprop).

Duas formas de esparsidade, ambas como descida de energia LOCAL (acréscimos ao laço
de inferência do PredictiveCoder, M4):
  • L1 (sparse coding de Olshausen & Field): penalidade β‖r‖₁, resolvida pelo
    operador proximal (soft-threshold). Esparsidade EMERGE da otimização.
  • k-WTA (k-winners-take-all): mantém só os k maiores r (inibição lateral cortical).
    Dá um ORÇAMENTO de atividade explícito e mensurável.

O aprendizado é o mesmo do M4 (ΔW ∝ ε·rᵀ), mas com r esparso só as COLUNAS ativas de
W mudam — D·k operações em vez de D·N. É a economia de "energia" que medimos.

MEDIÇÃO HONESTA (ver experiments/m22_sparse_efficient.py):
  A moeda é OPERAÇÃO SINÁPTICA (SynOps / MAC), uma contagem independente de hardware
  (o "AC op" da literatura neuromórfica), NÃO o relógio. Em Python/NumPy o laço esparso
  é até mais LENTO que o denso vetorizado (o M6 já mostrou isso); o que cai é o NÚMERO
  de operações sinápticas necessárias — a quantidade que hardware esparso/neuromórfico
  converte em energia. Isto prova o PRINCÍPIO de eficiência em miniatura; NÃO prova
  escala, nem wall-clock, nem eficiência biológica (estamos a ~9 ordens do cérebro).

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np

from .predictive import PredictiveCoder


class OpCounter:
    """Contador de operações sinápticas (MACs), denso vs dirigido a eventos.

    Separa as três contribuições da máquina de previsão e oferece duas leituras
    do termo bottom-up (Wᵀε), que é o ponto honesto: como ε é denso, recalcular o
    drive de TODAS as latentes custa D·N (`full`, o piso honesto); num substrato
    dirigido a eventos só as latentes ATIVAS recomputam seu drive (`warm`).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.macs_predict = 0          # x̂ = W r  (só unidades ativas propagam)
        self.macs_bottomup_full = 0    # Wᵀε denso (D·N por iteração)
        self.macs_bottomup_warm = 0    # Wᵀε dirigido a eventos (D·ativas)
        self.macs_update = 0           # ΔW = ε rᵀ (só colunas ativas)

    def total_full(self) -> int:
        return self.macs_predict + self.macs_bottomup_full + self.macs_update

    def total_warm(self) -> int:
        return self.macs_predict + self.macs_bottomup_warm + self.macs_update


class SparsePredictiveCoder(PredictiveCoder):
    """Predictive coder do M4 com código esparso (k-WTA e/ou L1) e contagem de SynOps.

    Compatível com `PredictiveCoder`: com `k_active=None` e `l1=0.0` o comportamento é
    IDÊNTICO ao M4 (teste de degeneração). Assim é drop-in nos agentes (M20/M21).
    """

    def __init__(self, n_obs: int, n_latent: int, n_infer: int = 30,
                 eta_r: float = 0.1, eta_w: float = 0.01, l2_prior: float = 0.1,
                 nonneg: bool = True, seed: int = 0,
                 k_active: int | None = None, l1: float = 0.0) -> None:
        super().__init__(n_obs, n_latent, n_infer=n_infer, eta_r=eta_r, eta_w=eta_w,
                         l2_prior=l2_prior, nonneg=nonneg, seed=seed)
        self.k_active = k_active
        self.l1 = l1

    # --- esparsificação do código (proximal do L1 + inibição lateral k-WTA) ---
    def _sparsify(self, r: np.ndarray) -> np.ndarray:
        if self.l1 > 0.0:                                  # soft-threshold (proximal L1)
            t = self.eta_r * self.l1
            r = np.sign(r) * np.maximum(np.abs(r) - t, 0.0)
        if self.nonneg:
            r = np.maximum(r, 0.0)
        if self.k_active is not None and self.n_latent > self.k_active:
            idx = np.argpartition(r, -self.k_active)[-self.k_active:]
            keep = np.zeros_like(r)
            keep[idx] = r[idx]                              # mantém só os k maiores
            r = keep
        return r

    def infer(self, x: np.ndarray, counter: OpCounter | None = None) -> np.ndarray:
        """Inferência esparsa (mesma descida de energia do M4 + esparsificação)."""
        D, N = self.n_obs, self.n_latent
        r = np.zeros(N)
        for it in range(self.n_infer):
            nnz = int(np.count_nonzero(r))                  # ativas que vão propagar
            err = x - self.W @ r                            # ε = x - x̂
            dr = self.W.T @ err - self.l2_prior * r         # -∂E/∂r
            if counter is not None:
                counter.macs_predict += D * nnz            # propagação só das ativas
                counter.macs_bottomup_full += D * N        # piso honesto (ε é denso)
                counter.macs_bottomup_warm += D * N if it == 0 else D * max(nnz, 1)
            r = r + self.eta_r * dr
            r = self._sparsify(r)
        return r

    def learn(self, x: np.ndarray, counter: OpCounter | None = None):
        """Infere r esparso e aplica ΔW ∝ ε·rᵀ só nas colunas ativas (regra local)."""
        r = self.infer(x, counter=counter)
        err = x - self.W @ r
        nz = np.nonzero(r)[0]
        if nz.size:
            self.W[:, nz] += self.eta_w * np.outer(err, r[nz])
        if counter is not None:
            counter.macs_predict += self.n_obs * nz.size    # previsão final
            counter.macs_update += self.n_obs * nz.size     # só colunas ativas
        self._normalize_columns()
        return r, float(np.sum(err ** 2))

    def prediction_error(self, x: np.ndarray) -> float:
        r = self.infer(x)
        err = x - self.W @ r
        return float(np.sum(err ** 2))

    def active_fraction(self, x: np.ndarray) -> float:
        """Fração de unidades latentes ativas (r>0) ao inferir x — a esparsidade."""
        r = self.infer(x)
        return float(np.count_nonzero(r)) / self.n_latent


# --- fórmulas fechadas de SynOps dos baselines densos (sem instrumentá-los) ---

def dense_learn_macs(n_obs: int, n_latent: int, n_infer: int) -> int:
    """SynOps de um learn() DENSO (todo neurônio, todo peso): M4 sem esparsidade.

    Por iteração de inferência: previsão W·r (D·N) + bottom-up Wᵀε (D·N).
    Mais a previsão final (D·N) e o update ΔW = ε·rᵀ (D·N).
    """
    return 2 * n_obs * n_latent * n_infer + 2 * n_obs * n_latent


def spiking_learn_ops(n_obs: int, n_latent: int, n_cycles: int, window: int) -> int:
    """Operações de um learn() do SpikingPredictiveCoder (M10), incluindo os passos LIF.

    Por ciclo: previsão (D·N) + bottom-up (D·N) sinápticos, e `window` passos de
    integração de membrana dos N neurônios LIF. Mais o update final (D·N).
    """
    syn = 2 * n_obs * n_latent * n_cycles + n_obs * n_latent
    neuron = n_cycles * window * n_latent
    return syn + neuron
