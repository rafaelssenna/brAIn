"""Mundo RICO: muitos objetos para o brAIn ter o que aprender de verdade — infra de escala.

Os mundos anteriores eram pequenos (6 posições): o agente saturava em ~20k passos, então
treino pesado não rendia. Este mundo é RICO: cada objeto combina QUATRO atributos —
  forma (barra, disco, cruz, anel),  cor,  tamanho,  posição (quadrante) —
gerando dezenas a centenas de objetos DISTINTOS, renderizados numa imagem COLORIDA 16×16×3
(D=768) com ruído realista. Agora há muito a perceber, distinguir e nomear, e um treino longo
de fato melhora o agente.

Cada atributo tem suas PALAVRAS (vocabulário grande e ancorado). O "professor" descreve a
cena por atributo (atenção conjunta), como no resto do projeto — nada de rótulos mágicos:
o agente forma seus conceitos dos pixels.

Honesto: continua sendo um mundo sintético de brinquedo (formas geométricas em grade), só que
MUITO maior em variedade. Não é o mundo real; é escala suficiente para o treino intensivo
render e para a percepção/linguagem terem o que crescer.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np

GRID = 16
N_CH = 3                       # canais de cor (R, G, B)
D = GRID * GRID * N_CH         # 768 — imagem colorida achatada

SHAPES = ["barra", "disco", "cruz", "anel"]
COLORS = ["vermelho", "verde", "azul", "amarelo"]
SIZES = ["pequeno", "medio", "grande"]
POSITIONS = ["topo-esq", "topo-dir", "base-esq", "base-dir"]

COLOR_RGB = {
    0: (1.0, 0.2, 0.2),   # vermelho
    1: (0.2, 1.0, 0.2),   # verde
    2: (0.3, 0.5, 1.0),   # azul
    3: (1.0, 0.9, 0.2),   # amarelo
}
SIZE_R = {0: 2, 1: 3, 2: 5}                      # raio por tamanho
QUAD = {0: (4, 4), 1: (4, 12), 2: (12, 4), 3: (12, 12)}   # centro de cada quadrante

ATTRS = ["forma", "cor", "tamanho", "posicao"]
ATTR_VALUES = [SHAPES, COLORS, SIZES, POSITIONS]
N_ATTR = len(ATTRS)


def n_objects():
    """Quantos objetos distintos o mundo tem (produto dos atributos)."""
    out = 1
    for v in ATTR_VALUES:
        out *= len(v)
    return out


def all_objects():
    """Lista de todas as combinações (forma, cor, tamanho, posição)."""
    objs = []
    for s in range(len(SHAPES)):
        for c in range(len(COLORS)):
            for sz in range(len(SIZES)):
                for p in range(len(POSITIONS)):
                    objs.append((s, c, sz, p))
    return objs


def _stamp(img, shape, cy, cx, r, rgb):
    """Desenha a forma `shape` centrada em (cy,cx) com raio r e cor rgb na imagem (GRID,GRID,3)."""
    yy, xx = np.ogrid[:GRID, :GRID]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if shape == 0:        # barra: linha horizontal de largura ~r
        mask = (np.abs(yy - cy) <= max(1, r // 2)) & (np.abs(xx - cx) <= r)
    elif shape == 1:      # disco: círculo cheio
        mask = dist <= r
    elif shape == 2:      # cruz: duas barras
        mask = (np.abs(yy - cy) <= 1) & (np.abs(xx - cx) <= r) | \
               (np.abs(xx - cx) <= 1) & (np.abs(yy - cy) <= r)
    else:                 # anel: círculo vazado
        mask = (dist <= r) & (dist >= max(1, r - 1.5))
    for ch in range(3):
        img[:, :, ch][mask] = rgb[ch]


def render(obj, rng=None, noise=0.05):
    """Renderiza um objeto (forma,cor,tamanho,posição) numa imagem colorida achatada (D,)."""
    s, c, sz, p = obj
    img = np.zeros((GRID, GRID, N_CH))
    cy, cx = QUAD[p]
    _stamp(img, s, cy, cx, SIZE_R[sz], COLOR_RGB[c])
    v = img.reshape(-1)
    if rng is not None:
        v = v + noise * rng.standard_normal(D)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------- vocabulário ancorado: palavras por atributo, em blocos disjuntos ----------
VOCAB = SHAPES + COLORS + SIZES + POSITIONS
WORD_OFFSET = np.cumsum([0] + [len(v) for v in ATTR_VALUES[:-1]]).tolist()
N_WORDS = len(VOCAB)


def word_of(attr, value):
    """Id global da palavra para (atributo, valor). Ex.: word_of(1, 0) = palavra 'vermelho'."""
    return WORD_OFFSET[attr] + value


def describe(obj):
    """As 4 palavras (uma por atributo) que descrevem o objeto."""
    return [word_of(a, obj[a]) for a in range(N_ATTR)]


def word_text(w):
    return VOCAB[w]


# ---------- o cérebro rico: um módulo especialista por atributo (como áreas corticais) ----------

class RichBrain:
    """Um cérebro para o mundo rico: um `LivingAgent` ESPECIALISTA por atributo (forma, cor,
    tamanho, posição) — como áreas corticais dedicadas. Cada módulo percebe a mesma imagem e
    aprende a nomear o SEU atributo. Junto, o cérebro descreve um objeto inteiro.

    Tem memória de longo prazo (save/load), então o treino pesado ACUMULA entre sessões.
    """

    def __init__(self, n_latent: int = 32, seed: int = 0):
        from .living import LivingAgent
        self.n_latent = n_latent
        self.modules = [LivingAgent(D, n_latent, len(ATTR_VALUES[a]), seed=1 + a + seed * 10)
                        for a in range(N_ATTR)]

    def learn(self, obj, rng):
        """Vive uma cena: percebe e ouve a palavra certa de CADA atributo (atenção conjunta)."""
        img = render(obj, rng)
        for a in range(N_ATTR):
            m = self.modules[a]
            c = m.concept(img)
            m.learn_perception(img)
            m.learn_listen(obj[a], c)
            m.reinforce_speak(c, obj[a], True)

    def describe(self, obj):
        """Vê o objeto e diz uma palavra por atributo (a descrição que o cérebro produz)."""
        img = render(obj)
        return [self.modules[a].speak(self.modules[a].concept(img), explore=False)
                for a in range(N_ATTR)]

    def naming_accuracy(self, objs):
        """Acurácia por atributo ao nomear um conjunto de objetos."""
        out = {}
        for a in range(N_ATTR):
            m = self.modules[a]
            out[ATTRS[a]] = float(np.mean([
                m.speak(m.concept(render(o)), explore=False) == o[a] for o in objs]))
        return out

    @property
    def lived(self):
        return self.modules[0]._t

    def save(self, path: str):
        """Salva os 4 módulos num diretório (memória de longo prazo do cérebro rico)."""
        import os
        os.makedirs(path, exist_ok=True)
        for a in range(N_ATTR):
            self.modules[a].save(os.path.join(path, f"mod_{ATTRS[a]}.npz"))

    @classmethod
    def load(cls, path: str):
        import os
        from .living import LivingAgent
        brain = cls.__new__(cls)
        brain.modules = [LivingAgent.load(os.path.join(path, f"mod_{ATTRS[a]}.npz"))
                         for a in range(N_ATTR)]
        brain.n_latent = brain.modules[0].nL
        return brain
