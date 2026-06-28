"""brAIn — núcleo do cérebro artificial.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.

Uma inteligência que nasce sem saber nada e aprende vivendo — paradigma
cerebral (neurônios spiking, plasticidade local, predictive coding,
embodiment), fora do paradigma Transformer/backprop.

M1: neurônio Leaky Integrate-and-Fire.
M2: sinapses plásticas com STDP.
M3: mundo e corpo (embodiment).
M4: predictive coding (a máquina de previsão).
M5: motivação intrínseca / curiosidade (desenvolvimento).
M7: organismo integrado (corpo + modelo de mundo + curiosidade).
M8: memória / replay (consolidação contra o esquecimento).
M9: predictive coding hierárquico (rumo aos conceitos).
M10: predictive coding spiking (substrato unificado).
M11: agência / planejamento (o primeiro "pensar").
M13: organismo integrado (a costura final — tudo num laço só).
M14: previsão temporal/sequencial (Fase 3 — rumo à cognição).
M15: gramática (regras sequenciais). M16: comunicação emergente.
M17: composicionalidade. M18: símbolos ancorados na percepção.
M19: a língua nasce da vivência. M20: o organismo vivo (a costura da cognição).
M21: palavras de português ancoradas na percepção.
M22: substrato esparso e dirigido a eventos (método > força bruta).
M23: o organismo vivo rodando no substrato esparso (mesma cognição, menos energia).
M24: o organismo vivo percebendo com neurônios que disparam (a cognição sobrevive aos spikes).
M25: o organismo corporificado que fala (corpo + percepção + curiosidade + memória + linguagem num laço).
M26: o organismo que planeja rotas num mundo 2D e fala (planejar + contornar paredes + nomear o que alcança).
"""

from .neuron import LIFPopulation, fi_curve_theory
from .synapse import STDPConnection
from .world import LightWorld, Vehicle
from .predictive import PredictiveCoder
from .curiosity import IntrinsicMotivation, select_region
from .agent import RingWorld, IntegratedAgent
from .memory import ReplayBuffer
from .hierarchy import HierarchicalPredictiveCoder
from .spiking_predictive import SpikingPredictiveCoder, SpikingPerceptionCoder
from .sparse_predictive import SparsePredictiveCoder, OpCounter, dense_learn_macs, spiking_learn_ops
from .planning import (GridWorld, WorldModel, explore_and_learn, plan,
                       run_planner, run_reactive)
from .integrated import IntegratedBrain
from .temporal import TemporalPredictiveCoder
from .communication import SignalingGame, CompositionalGame, GroundedLanguageGame, LivedLanguageGame
from .living import LivingAgent
from .embodied_language import EmbodiedLanguageAgent, communication_accuracy
from .planning_language import PlanningLanguageAgent

__all__ = [
    "TemporalPredictiveCoder", "SignalingGame", "CompositionalGame", "GroundedLanguageGame",
    "LivedLanguageGame",
    "LIFPopulation", "fi_curve_theory", "STDPConnection", "LightWorld", "Vehicle",
    "PredictiveCoder", "IntrinsicMotivation", "select_region",
    "RingWorld", "IntegratedAgent", "ReplayBuffer", "HierarchicalPredictiveCoder",
    "SpikingPredictiveCoder", "SpikingPerceptionCoder",
    "SparsePredictiveCoder", "OpCounter", "dense_learn_macs", "spiking_learn_ops",
    "GridWorld", "WorldModel", "explore_and_learn", "plan", "run_planner", "run_reactive",
    "IntegratedBrain", "LivingAgent",
    "EmbodiedLanguageAgent", "communication_accuracy",
    "PlanningLanguageAgent",
]
