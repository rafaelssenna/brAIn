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
"""

from .neuron import LIFPopulation, fi_curve_theory
from .synapse import STDPConnection
from .world import LightWorld, Vehicle
from .predictive import PredictiveCoder
from .curiosity import IntrinsicMotivation, select_region
from .agent import RingWorld, IntegratedAgent
from .memory import ReplayBuffer
from .hierarchy import HierarchicalPredictiveCoder
from .spiking_predictive import SpikingPredictiveCoder
from .planning import (GridWorld, WorldModel, explore_and_learn, plan,
                       run_planner, run_reactive)
from .integrated import IntegratedBrain
from .temporal import TemporalPredictiveCoder
from .communication import SignalingGame, CompositionalGame, GroundedLanguageGame

__all__ = [
    "TemporalPredictiveCoder", "SignalingGame", "CompositionalGame", "GroundedLanguageGame",
    "LIFPopulation", "fi_curve_theory", "STDPConnection", "LightWorld", "Vehicle",
    "PredictiveCoder", "IntrinsicMotivation", "select_region",
    "RingWorld", "IntegratedAgent", "ReplayBuffer", "HierarchicalPredictiveCoder",
    "SpikingPredictiveCoder",
    "GridWorld", "WorldModel", "explore_and_learn", "plan", "run_planner", "run_reactive",
    "IntegratedBrain",
]
