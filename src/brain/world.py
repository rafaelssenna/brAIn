"""Mundo e corpo — M3 (embodiment).

Um ambiente mínimo para fechar o laço percepção -> ação -> percepção.

  - `LightWorld`: arena 2D com uma fonte (luz/comida). Fornece a ativação dos
    sensores do agente dada sua pose (posição + orientação).
  - `Vehicle`: um corpo de tração diferencial (duas rodas). Recebe velocidade
    de cada roda e atualiza posição/orientação (cinemática de Braitenberg).

O "cérebro" (neurônios + sinapses) NÃO está aqui — ele é montado no
experimento, ligando sensores a motores. Assim o corpo é reaproveitável.

Convenção de ângulos: orientação `heading` em radianos, sentido anti-horário,
0 apontando para +x.
"""

from __future__ import annotations

import numpy as np


class LightWorld:
    """Arena quadrada [0, size]^2 com uma fonte pontual."""

    def __init__(self, size: float = 100.0, source: tuple[float, float] | None = None,
                 falloff: float = 40.0, sensor_offset: float = 0.6) -> None:
        self.size = size
        self.source = np.array(source if source is not None else (size / 2, size / 2),
                                dtype=np.float64)
        self.falloff = falloff          # escala de queda da intensidade com a distância
        self.sensor_offset = sensor_offset  # ângulo (rad) dos olhos esq/dir vs heading

    def intensity(self, dist: float) -> float:
        """Intensidade da fonte a uma distância `dist` (cai com 1/(1+(d/D0)^2))."""
        return 1.0 / (1.0 + (dist / self.falloff) ** 2)

    def sensor_activations(self, pos: np.ndarray, heading: float):
        """Ativação dos dois sensores (esquerdo, direito) em [0, 1] + distância.

        Cada sensor "vê" mais quanto (a) mais perto a fonte e (b) mais o olho
        aponta para ela. O olho esquerdo aponta para `heading + offset`, o
        direito para `heading - offset`.
        """
        d = self.source - pos
        dist = float(np.hypot(d[0], d[1]))
        ang_to_source = float(np.arctan2(d[1], d[0]))
        inten = self.intensity(dist)

        left_dir = heading + self.sensor_offset
        right_dir = heading - self.sensor_offset
        left = inten * max(0.0, np.cos(ang_to_source - left_dir))
        right = inten * max(0.0, np.cos(ang_to_source - right_dir))
        return left, right, dist


class Vehicle:
    """Corpo de tração diferencial (duas rodas)."""

    def __init__(self, pos, heading: float = 0.0, wheel_base: float = 6.0,
                 size: float = 100.0) -> None:
        self.pos = np.array(pos, dtype=np.float64)
        self.heading = float(heading)
        self.wheel_base = wheel_base
        self.size = size

    def step(self, v_left: float, v_right: float, dt: float = 1.0) -> None:
        """Atualiza pose dadas as velocidades das rodas esquerda/direita.

        forward = média das rodas; rotação = diferença / distância entre rodas.
        Roda direita mais rápida => vira à esquerda (anti-horário, +heading).
        """
        forward = 0.5 * (v_left + v_right)
        dtheta = (v_right - v_left) / self.wheel_base
        self.heading += dtheta * dt
        self.pos = self.pos + forward * dt * np.array(
            [np.cos(self.heading), np.sin(self.heading)]
        )
        # Mantém o corpo dentro da arena.
        self.pos = np.clip(self.pos, 0.0, self.size)

    def distance_to(self, target: np.ndarray) -> float:
        return float(np.hypot(*(target - self.pos)))
