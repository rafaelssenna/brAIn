"""Conversa emergente com FRASES — M28 (do descrever ao conversar).

O M27 aprendeu a primeira frase ouvindo um PROFESSOR (que já sabia a resposta certa).
Mas conversar é dois cérebros se entendendo SEM ninguém dizer a resposta — coordenando do
zero, como dois bebês (ou duas pessoas sem língua comum) inventando uma língua. Este marco
dá o passo: dois agentes inventam, juntos, uma mini-língua com FRASES de duas palavras
(uma para a FORMA, outra para a POSIÇÃO) só para se entenderem sobre cenas.

Sem professor: o falante vê a cena e produz uma frase; o ouvinte decodifica e tenta
identificar a cena. Acertou -> ambos reforçam; errou -> enfraquecem. O ouvinte aprende o
pareamento verdadeiro por ATENÇÃO CONJUNTA (vê a cena referida — o que quebra o deadlock de
coordenação, como no M16). Ninguém programa o vocabulário NEM a ordem das palavras.

Pergunta científica (H11): quando dois cérebros PRECISAM se coordenar, emerge uma língua
com FRASES — vocabulário E uma ORDEM de palavras consistente entre eles? E o código
COMPOSICIONAL (símbolo por atributo) GENERALIZA para cenas que eles nunca disseram?

  mode='compositional': cada slot da frase carrega UM atributo (forma OU posição) — pode
                        generalizar para combinações novas (produtividade da sintaxe).
  mode='holistic':      a frase inteira é memorizada por cena — decora, não generaliza.

Honesto: escala de brinquedo (frase de 2 palavras, poucos atributos, tamanho fixo). É o
MECANISMO da conversa composicional emergente, NÃO linguagem plena (sem recursão, sem
tamanho variável, sem semântica profunda) — horizonte distante, provável híbrido.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np


class _Agent:
    """Um agente que fala e ouve frases de 2 slots, compondo por atributo (ou holístico)."""

    def __init__(self, k_shape, k_pos, n_words, mode, lr, temp, init_noise, rng):
        self.k_shape, self.k_pos = k_shape, k_pos
        self.W = n_words
        self.mode = mode
        self.lr = lr
        self.temp = temp
        self.rng = rng
        rn = lambda *s: init_noise * rng.standard_normal(s)
        if mode == "compositional":
            # produção: por slot, atributo -> palavra (forma e posição competem pelo slot)
            self.Ssh = rn(2, k_shape, n_words)        # [slot, forma]   -> palavra
            self.Spo = rn(2, k_pos, n_words)          # [slot, posição] -> palavra
            # compreensão: por slot, palavra -> atributo (de cada papel)
            self.Rsh = rn(2, n_words, k_shape)        # [slot, palavra] -> forma
            self.Rpo = rn(2, n_words, k_pos)          # [slot, palavra] -> posição
            # qual papel cada slot carrega (emerge da coordenação)
            self.role_ev = np.zeros((2, 2))           # [slot, (shape=0,pos=1)]
        else:                                         # holístico
            nc = k_shape * k_pos
            self.Sf = [rn(nc, n_words), rn(nc, n_words)]   # cena -> palavra por slot
            self.Rf = rn(n_words * n_words, nc)            # par de palavras -> cena

    def _sm(self, v):
        z = np.exp((v - v.max()) / self.temp); return z / z.sum()

    def _pick(self, prefs, explore):
        return int(self.rng.choice(len(prefs), p=self._sm(prefs))) if explore else int(np.argmax(prefs))

    def slot_role(self, slot):
        """Papel aprendido do slot, com EXCLUSIVIDADE MÚTUA: os dois slots competem pelos
        papéis (um vira 'shape', o outro 'pos'). O slot com maior preferência relativa por
        forma fica com 'shape'; o outro, com 'pos'. Assim a frase nunca colapsa em dois
        papéis iguais (ex.: 'pos','pos') — a ordem emerge como uma divisão de trabalho."""
        # preferência de cada slot por forma vs posição (diferença das evidências)
        pref0 = self.role_ev[0, 0] - self.role_ev[0, 1]    # slot 0: + => tende a forma
        pref1 = self.role_ev[1, 0] - self.role_ev[1, 1]    # slot 1
        shape_slot = 0 if pref0 >= pref1 else 1            # quem prefere mais forma fica com ela
        return "shape" if slot == shape_slot else "pos"

    # ---------- falar: vê a cena, produz a frase de 2 palavras ----------
    def speak(self, shape, pos, explore=True):
        if self.mode == "compositional":
            words = [0, 0]
            for slot in range(2):
                if self.slot_role(slot) == "shape":
                    words[slot] = self._pick(self.Ssh[slot, shape], explore)
                else:
                    words[slot] = self._pick(self.Spo[slot, pos], explore)
            return words
        else:
            c = shape * self.k_pos + pos
            return [self._pick(self.Sf[0][c], explore), self._pick(self.Sf[1][c], explore)]

    # ---------- ouvir: decodifica a frase em (forma, posição) ----------
    def understand(self, words):
        if self.mode == "compositional":
            sh = po = None
            for slot in range(2):
                if self.slot_role(slot) == "shape":
                    sh = int(np.argmax(self.Rsh[slot, words[slot]]))
                else:
                    po = int(np.argmax(self.Rpo[slot, words[slot]]))
            if sh is None:
                sh = int(np.argmax(self.Rsh[0, words[0]]))
            if po is None:
                po = int(np.argmax(self.Rpo[1, words[1]]))
            return sh, po
        else:
            pair = words[0] * self.W + words[1]
            c = int(np.argmax(self.Rf[pair]))
            return c // self.k_pos, c % self.k_pos

    # ---------- aprender (sem professor: reforço por sucesso + atenção conjunta) ----------
    def learn_speak(self, shape, pos, words, reward):
        d = self.lr if reward else -self.lr
        if self.mode == "compositional":
            for slot in range(2):
                if self.slot_role(slot) == "shape":
                    self.Ssh[slot, shape, words[slot]] += d
                else:
                    self.Spo[slot, pos, words[slot]] += d
        else:
            c = shape * self.k_pos + pos
            self.Sf[0][c, words[0]] += d; self.Sf[1][c, words[1]] += d

    def learn_listen(self, words, shape, pos):
        """Atenção conjunta: vê a cena verdadeira e aprende palavra->atributo por slot.
        Também acumula evidência de qual papel cada slot carrega (a ordem emerge)."""
        if self.mode == "compositional":
            for slot in range(2):
                self.Rsh[slot, words[slot], shape] += self.lr
                self.Rpo[slot, words[slot], pos] += self.lr
            # evidência de papel: o slot cujo palavra prediz melhor a forma "vira" shape.
            # heurística simples e local: o slot 0 tende a forma se sua palavra a separa mais.
            self._update_roles(words, shape, pos)
        else:
            c = shape * self.k_pos + pos
            self.Rf[words[0] * self.W + words[1], c] += self.lr

    def _update_roles(self, words, shape, pos):
        """Acumula, por slot, o quanto a palavra dali está associada a forma vs posição.
        O papel emerge: o slot com maior associação concentrada em forma vira 'shape'."""
        for slot in range(2):
            sh_strength = self.Rsh[slot, words[slot]].max() - self.Rsh[slot, words[slot]].mean()
            po_strength = self.Rpo[slot, words[slot]].max() - self.Rpo[slot, words[slot]].mean()
            self.role_ev[slot, 0] += sh_strength
            self.role_ev[slot, 1] += po_strength


class EmergentSentenceGame:
    """Dois agentes inventam uma língua com frases (forma×posição) sem professor.

    Em cada rodada, um agente (alternando) é o FALANTE: vê uma cena, produz a frase; o
    outro é o OUVINTE: decodifica e adivinha a cena. Recompensa = acertou a cena inteira.
    Ambos aprendem (falante por reforço, ouvinte por atenção conjunta). A língua e a ORDEM
    das palavras emergem da necessidade de se coordenar.
    """

    def __init__(self, k_shape: int, k_pos: int, n_words: int | None = None,
                 mode: str = "compositional", lr: float = 0.2, temp: float = 0.4,
                 init_noise: float = 0.05, seed: int = 0):
        self.k_shape, self.k_pos = k_shape, k_pos
        self.mode = mode
        self.W = n_words or (max(k_shape, k_pos) + 1)
        self.rng = np.random.default_rng(seed)
        self.A = _Agent(k_shape, k_pos, self.W, mode, lr, temp, init_noise, self.rng)
        self.B = _Agent(k_shape, k_pos, self.W, mode, lr, temp, init_noise, self.rng)

    def play(self, shape, pos, speaker_is_A=True, learn=True, explore=True):
        """Uma rodada de conversa sobre a cena (shape,pos). Retorna 1 se se entenderam."""
        spk, lis = (self.A, self.B) if speaker_is_A else (self.B, self.A)
        words = spk.speak(shape, pos, explore=explore)
        gs, gp = lis.understand(words)
        reward = int(gs == shape and gp == pos)
        if learn:
            spk.learn_speak(shape, pos, words, reward)
            lis.learn_listen(words, shape, pos)            # atenção conjunta (vê a cena)
        return reward

    def accuracy(self, scenes):
        """Acurácia determinística nos dois sentidos (A->B e B->A) sobre as cenas dadas."""
        tot = 0.0
        for (s, p) in scenes:
            tot += self.play(s, p, speaker_is_A=True, learn=False, explore=False)
            tot += self.play(s, p, speaker_is_A=False, learn=False, explore=False)
        return tot / (2 * len(scenes))

    def emergent_order(self):
        """A ordem de palavras que emergiu para cada agente (papel de cada slot)."""
        return (tuple(self.A.slot_role(s) for s in range(2)),
                tuple(self.B.slot_role(s) for s in range(2)))
