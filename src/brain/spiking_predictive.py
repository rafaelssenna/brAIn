"""Predictive coding SPIKING — M10 (Fase 2, o salto purista).

Costura M1 (neurônio LIF) + M4 (predictive coding) + plasticidade local numa
rede só: as unidades de representação são neurônios LIF REAIS; a TAXA de disparo
deles carrega o valor da causa latente.

Laço (código de taxa):
  1. corrente nos neurônios r = ganho · (Wᵀ ε)   (erro de previsão, bottom-up)
  2. roda os LIF por uma janela; lê a taxa de disparo ρ (∈ [0,1] normalizada)
  3. previsão x̂ = W ρ;  erro ε = x - x̂;  repete alguns ciclos até assentar
  4. aprende, LOCAL:  ΔW ∝ ε ρᵀ   (Hebbiano sobre o erro e a taxa de spikes)

Observações honestas:
  • A não-negatividade sai DE GRAÇA: corrente negativa => neurônio não dispara
    => ρ=0 (retificação pelo limiar do spike). Mesmo papel do ReLU no M4.
  • É *rate-coded*: o erro é lido em taxa, não em spikes temporais puros (o
    spiking predictive coding temporal completo é mais difícil — trabalho futuro).
  • Pequeno e lento (laço spiking em Python). É prova de conceito do substrato.
"""

from __future__ import annotations

import numpy as np

from .neuron import LIFPopulation
from .sparse_predictive import OpCounter


class SpikingPredictiveCoder:
    """Predictive coder de uma camada cujas unidades latentes são neurônios LIF."""

    def __init__(self, n_obs: int, n_latent: int, n_cycles: int = 14,
                 window: int = 50, dt: float = 1.0, in_gain: float = 25.0,
                 infer_lr: float = 0.3, eta_w: float = 0.04, l2_prior: float = 0.05,
                 seed: int = 0):
        self.n_obs = n_obs
        self.n_latent = n_latent
        self.n_cycles = n_cycles
        self.window = window
        self.in_gain = in_gain
        self.infer_lr = infer_lr        # amortecimento da inferência (integrador c/ vazamento)
        self.eta_w = eta_w
        self.l2_prior = l2_prior

        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.1, size=(n_obs, n_latent))
        self._normalize()

        self.lif = LIFPopulation(n=n_latent, dt=dt)
        # Máximo aproximado de spikes na janela (limitado pelo período refratário).
        self.max_count = window / 2.0

    def _normalize(self):
        self.W = self.W / (np.linalg.norm(self.W, axis=0, keepdims=True) + 1e-8)

    def _spike_rates(self, current: np.ndarray) -> np.ndarray:
        """Roda os LIF por uma janela com corrente constante; taxa normalizada."""
        self.lif.reset_state()
        counts = np.zeros(self.n_latent)
        for _ in range(self.window):
            counts += self.lif.step(current)
        return np.clip(counts / self.max_count, 0.0, 1.0)

    def infer(self, x: np.ndarray):
        """Assenta a representação via ciclos de atividade spiking. Retorna (ρ, ε)."""
        rho = np.zeros(self.n_latent)
        for _ in range(self.n_cycles):
            err = x - self.W @ rho
            current = self.in_gain * (self.W.T @ err) - self.l2_prior * rho
            target = self._spike_rates(current)     # spikes; corrente<0 => silêncio
            # integrador com vazamento: acumula em vez de substituir (evita oscilar)
            rho = (1.0 - self.infer_lr) * rho + self.infer_lr * target
        err = x - self.W @ rho
        return rho, err

    def learn(self, x: np.ndarray):
        """Infere e aplica a regra LOCAL ΔW ∝ ε ρᵀ (sobre a taxa de spikes)."""
        rho, err = self.infer(x)
        self.W += self.eta_w * np.outer(err, rho)
        self._normalize()
        return rho, float(np.sum(err ** 2))

    def prediction_error(self, x: np.ndarray) -> float:
        _, err = self.infer(x)
        return float(np.sum(err ** 2))

    def spike_raster(self, x: np.ndarray):
        """Para visualização: tempos de disparo dos neurônios r sob a entrada x
        (na corrente já assentada). Retorna lista de arrays de tempos (ms)."""
        rho, _ = self.infer(x)
        err = x - self.W @ rho
        current = self.in_gain * (self.W.T @ err) - self.l2_prior * rho
        self.lif.reset_state()
        times = [[] for _ in range(self.n_latent)]
        for k in range(self.window):
            sp = self.lif.step(current)
            for i in np.flatnonzero(sp):
                times[i].append(k * self.lif.dt)
        return [np.array(t) for t in times]


class SpikingPerceptionCoder(SpikingPredictiveCoder):
    """Adaptador do M10 para ser DROP-IN no organismo vivo (M20) — M24.

    O `LivingAgent` (M20/M21) e o experimento esperam um preditor com a mesma API
    do M4/M22: `infer(x) -> r` (só o vetor latente), `learn(x) -> (r, sse)`,
    `prediction_error(x)`, `active_fraction(x)` e contagem de SynOps via `OpCounter`.
    O `SpikingPredictiveCoder` original tem um `infer` que devolve `(rho, err)` — esta
    subclasse só ajusta a ASSINATURA (sem mudar a matemática do substrato spiking) para
    encaixar no organismo, do mesmo jeito que o `SparsePredictiveCoder` (M22) encaixou.

    A esparsidade aqui não é imposta (k-WTA): EMERGE do limiar do spike — corrente
    sublimiar => ρ=0 (o neurônio não dispara). É a retificação biológica de graça.
    """

    def infer(self, x: np.ndarray, counter: OpCounter | None = None) -> np.ndarray:
        """Inferência spiking, mas retornando só ρ (a taxa de disparo latente).

        Mantém o laço do M10 (ciclos de atividade LIF) e, se `counter` for dado,
        contabiliza as operações sinápticas (previsão W·ρ e bottom-up Wᵀε por ciclo)
        e os passos de integração de membrana dos neurônios LIF — a 'energia' do
        substrato spiking, na mesma moeda (SynOps) do M22.
        """
        D, N = self.n_obs, self.n_latent
        rho = np.zeros(N)
        for _ in range(self.n_cycles):
            err = x - self.W @ rho
            current = self.in_gain * (self.W.T @ err) - self.l2_prior * rho
            target = self._spike_rates(current)
            rho = (1.0 - self.infer_lr) * rho + self.infer_lr * target
            if counter is not None:
                nnz = int(np.count_nonzero(rho))
                counter.macs_predict += D * max(nnz, 1)     # W·ρ (só ativos propagam)
                counter.macs_bottomup_full += D * N         # Wᵀε denso (ε é denso)
                counter.macs_bottomup_warm += D * N         # spiking recomputa todos
                counter.macs_update += N * self.window      # passos de membrana LIF
        self._last_err = x - self.W @ rho
        return rho

    def learn(self, x: np.ndarray, counter: OpCounter | None = None):
        """Infere ρ (spiking) e aplica a regra LOCAL ΔW ∝ ε ρᵀ (Hebbiano sobre o erro)."""
        rho = self.infer(x, counter=counter)
        err = x - self.W @ rho
        self.W += self.eta_w * np.outer(err, rho)
        self._normalize()
        if counter is not None:
            counter.macs_update += self.n_obs * int(np.count_nonzero(rho))
        return rho, float(np.sum(err ** 2))

    def prediction_error(self, x: np.ndarray) -> float:
        rho = self.infer(x)
        return float(np.sum((x - self.W @ rho) ** 2))

    def active_fraction(self, x: np.ndarray) -> float:
        """Fração de neurônios que disparam (ρ>0) ao perceber x — a esparsidade emergente."""
        rho = self.infer(x)
        return float(np.count_nonzero(rho)) / self.n_latent

    def spike_raster(self, x: np.ndarray):
        """Raster (tempos de disparo por neurônio) ao perceber x. Reusa a lógica do M10,
        mas com o `infer` desta subclasse (que devolve só ρ) — para a figura do M24."""
        rho = self.infer(x)
        err = x - self.W @ rho
        current = self.in_gain * (self.W.T @ err) - self.l2_prior * rho
        self.lif.reset_state()
        times = [[] for _ in range(self.n_latent)]
        for k in range(self.window):
            sp = self.lif.step(current)
            for i in np.flatnonzero(sp):
                times[i].append(k * self.lif.dt)
        return [np.array(t) for t in times]
