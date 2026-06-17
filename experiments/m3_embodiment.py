"""M3 — Corpo e mundo (embodiment).

Fecha o laço percepção -> ação -> percepção. Um agente spiking corporificado
faz FOTOTAXIA (busca a fonte) com fiação reflexiva estilo Braitenberg:

    olho ESQ --(+, fixa)--> motor DIR     (cruzado => vira para a fonte)
    olho DIR --(+, fixa)--> motor ESQ
    proximidade --(-, fixa)--> ambos motores  (freia ao chegar => não atravessa)

Comportamento composto, tudo de fiação fixa (sem aprendizado — isso é M4/M5):
  - VARREDURA (klinocinese): viés tônico assimétrico nos motores faz o agente
    girar devagar quando está no escuro, varrendo o campo de visão até detectar
    a fonte (resolve o ponto cego traseiro dos olhos, que só veem à frente).
  - FOTOTAXIA: ao detectar a fonte, a excitação cruzada domina e vira para ela.
  - CHEGADA: perto da fonte, o neurônio de proximidade inibe os motores e o
    agente PARA (a inibição cancela inclusive o viés de varredura).

Objetivo do M3: provar que o laço corporificado FUNCIONA e BATE o baseline
aleatório (regra "baseline sempre").

Uso:   python experiments/m3_embodiment.py
Salva: experiments/output/m3_embodiment.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brain import LIFPopulation, STDPConnection, LightWorld, Vehicle  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
ARENA = 100.0
REACH_RADIUS = 8.0

# Configuração do agente (tunável; valores escolhidos por varredura em _m3_sweep.py).
DEFAULT = dict(
    dt=1.0,
    ctrl_window=40,       # passos neurais (ms) por passo de física
    n_ctrl_steps=250,     # passos de física por episódio
    i_bias=1.2,           # corrente de base dos sensores (abaixo do limiar)
    i_gain=4.0,           # ganho sensor -> corrente
    prox_thresh=0.92,     # intensidade a partir da qual a proximidade "acorda"
    w_exc=3.0,            # peso excitatório olho -> motor (cruzado)
    w_inh=6.0,            # peso inibitório (módulo) proximidade -> motores
    motor_bias_l=1.8,     # viés tônico do motor esquerdo (varredura)
    motor_bias_r=2.2,     # viés tônico do motor direito (assimetria => gira)
    v_k=0.9,              # disparos do motor na janela -> velocidade da roda
)


def build_brain(cfg):
    """3 sensoriais -> 2 motores, fiação fixa (excitatória cruzada + inibição)."""
    sensory = LIFPopulation(n=3, dt=cfg["dt"])   # 0=olho ESQ, 1=olho DIR, 2=proximidade
    motor = LIFPopulation(n=2, dt=cfg["dt"])     # 0=motor ESQ, 1=motor DIR
    w = np.zeros((3, 2))
    w[0, 1] = cfg["w_exc"]    # olho ESQ -> motor DIR
    w[1, 0] = cfg["w_exc"]    # olho DIR -> motor ESQ
    w[2, 0] = -cfg["w_inh"]   # proximidade -> motor ESQ (freio)
    w[2, 1] = -cfg["w_inh"]   # proximidade -> motor DIR (freio)
    conn = STDPConnection(3, 2, w_init=w, plastic=False, tau_syn=10.0, dt=cfg["dt"])
    return sensory, motor, conn


def run_braitenberg(world, start_pos, start_heading, cfg=DEFAULT):
    """Roda um episódio do agente spiking. Retorna (trajetória, distâncias)."""
    sensory, motor, conn = build_brain(cfg)
    body = Vehicle(start_pos, start_heading, size=world.size)
    motor_bias = np.array([cfg["motor_bias_l"], cfg["motor_bias_r"]])

    traj = [body.pos.copy()]
    dists = [body.distance_to(world.source)]

    for _ in range(cfg["n_ctrl_steps"]):
        left, right, dist = world.sensor_activations(body.pos, body.heading)
        prox = np.clip(
            (world.intensity(dist) - cfg["prox_thresh"]) / (1.0 - cfg["prox_thresh"]),
            0.0, 1.0)
        i_sens = np.array([cfg["i_bias"] + cfg["i_gain"] * left,
                           cfg["i_bias"] + cfg["i_gain"] * right,
                           cfg["i_bias"] + cfg["i_gain"] * prox])

        motor_count = np.zeros(2)
        for _ in range(cfg["ctrl_window"]):
            s_spk = sensory.step(i_sens)
            i_motor = conn.propagate(s_spk) + motor_bias  # viés tônico = varredura
            m_spk = motor.step(i_motor)
            motor_count += m_spk

        # Velocidade da roda = ganho * disparos. Sem termo fixo: a inibição de
        # proximidade pode zerar os motores => o agente PARA de verdade.
        v_left = cfg["v_k"] * motor_count[0]
        v_right = cfg["v_k"] * motor_count[1]
        body.step(v_left, v_right)

        traj.append(body.pos.copy())
        dists.append(body.distance_to(world.source))
    return np.array(traj), np.array(dists)


def run_random(world, start_pos, start_heading, rng, cfg=DEFAULT):
    """Baseline: anda para frente com mudanças de direção aleatórias."""
    body = Vehicle(start_pos, start_heading, size=world.size)
    traj = [body.pos.copy()]
    dists = [body.distance_to(world.source)]
    speed = 3.0  # velocidade comparável à do agente em homing
    for _ in range(cfg["n_ctrl_steps"]):
        body.heading += rng.uniform(-0.5, 0.5)
        body.step(speed, speed)
        traj.append(body.pos.copy())
        dists.append(body.distance_to(world.source))
    return np.array(traj), np.array(dists)


def evaluate(cfg=DEFAULT, n_trials=40, seed=SEED):
    """Roda n_trials com fonte/partida aleatórias. Retorna métricas + curvas."""
    rng = np.random.default_rng(seed)
    b_curves, r_curves = [], []
    b_reached, r_reached = 0, 0
    for _ in range(n_trials):
        src = rng.uniform(20, 80, size=2)
        start = rng.uniform(0, ARENA, size=2)
        heading = rng.uniform(-np.pi, np.pi)
        w = LightWorld(size=ARENA, source=tuple(src))
        _, db = run_braitenberg(w, start, heading, cfg)
        _, dr = run_random(w, start, heading, rng, cfg)
        b_curves.append(db); r_curves.append(dr)
        b_reached += int(db.min() < REACH_RADIUS)
        r_reached += int(dr.min() < REACH_RADIUS)
    b_curves = np.array(b_curves); r_curves = np.array(r_curves)
    return dict(
        b_curves=b_curves, r_curves=r_curves,
        b_success=100 * b_reached / n_trials,
        r_success=100 * r_reached / n_trials,
        b_final=b_curves[:, -1].mean(), r_final=r_curves[:, -1].mean(),
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # (a) Trajetórias de exemplo: fonte central, partidas variadas (inclui de costas).
    world = LightWorld(size=ARENA, source=(50.0, 50.0))
    starts = [((10, 10), np.pi), ((90, 15), np.pi / 2),     # 2 começam "de costas"
              ((85, 88), 0.0), ((12, 85), -np.pi / 2)]
    braiten_trajs = [run_braitenberg(world, p, h)[0] for p, h in starts]
    rand_traj = run_random(world, (10, 10), 0.0, np.random.default_rng(SEED))[0]

    # (b)/(c) Estatística.
    m = evaluate(DEFAULT, n_trials=40)
    b, r = m["b_curves"], m["r_curves"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for tr in braiten_trajs:
        ax.plot(tr[:, 0], tr[:, 1], lw=1.3, color="#2ca02c", alpha=0.9)
        ax.scatter(tr[0, 0], tr[0, 1], s=20, color="#2ca02c", zorder=4)  # partida
    ax.plot(rand_traj[:, 0], rand_traj[:, 1], lw=1.0, color="#888888",
            alpha=0.8, label="aleatório")
    ax.plot([], [], color="#2ca02c", label="brAIn (• = partida)")
    ax.scatter([50], [50], marker="*", s=320, color="#ffb000",
               edgecolor="k", zorder=5, label="fonte")
    ax.set_title("(a) Trajetórias: varre, acha, chega e para")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(0, ARENA); ax.set_ylim(0, ARENA)
    ax.legend(loc="upper right", fontsize=7); ax.set_aspect("equal")

    ax = axes[1]
    t = np.arange(b.shape[1])
    for curves, color, lbl in [(b, "#2ca02c", "brAIn"), (r, "#888888", "aleatório")]:
        mn = curves.mean(0); sd = curves.std(0)
        ax.plot(t, mn, lw=1.8, color=color, label=lbl)
        ax.fill_between(t, mn - sd, mn + sd, color=color, alpha=0.18)
    ax.axhline(REACH_RADIUS, ls="--", lw=0.8, color="#d62728", label="raio de chegada")
    ax.set_title("(b) Distância à fonte (média de 40 episódios)")
    ax.set_xlabel("passo de controle"); ax.set_ylabel("distância à fonte")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    rates = [m["b_success"], m["r_success"]]
    bars = ax.bar(["brAIn", "aleatório"], rates, color=["#2ca02c", "#888888"])
    for bar, rt in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rt + 1.5, f"{rt:.0f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_title("(c) Taxa de sucesso (chegou à fonte)")
    ax.set_ylabel("% de episódios"); ax.set_ylim(0, 105)

    fig.suptitle("brAIn · M3 — Embodiment: o laço percepção→ação→percepção fecha",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = os.path.join(OUT_DIR, "m3_embodiment.png")
    fig.savefig(out_path, dpi=130)
    print(f"Figura salva em: {out_path}")
    print(f"Taxa de sucesso  brAIn: {m['b_success']:.0f}%  |  aleatório: {m['r_success']:.0f}%")
    print(f"Distância final média  brAIn: {m['b_final']:.1f}  |  aleatório: {m['r_final']:.1f}")


if __name__ == "__main__":
    main()
