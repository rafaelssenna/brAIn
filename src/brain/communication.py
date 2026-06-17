"""Comunicação emergente — M16 (Fase 3): o primeiro "conversar".

Dois agentes inventam, do zero, um código compartilhado símbolo<->conceito,
jogando um JOGO DE REFERÊNCIA (signaling game, Lewis):
  • o EMISSOR vê um conceito e emite um símbolo;
  • o RECEPTOR ouve o símbolo e adivinha qual era o conceito;
  • se acertam, ambos REFORÇAM a associação usada; se erram, enfraquecem.

O aprendizado é modulado por recompensa (Hebbiano de três fatores — a recompensa
é o "terceiro fator"). Ninguém programa o vocabulário: ele EMERGE da necessidade
de coordenar. Os conceitos são ancorados (os mesmos que o brAIn forma no M9), então
é comunicação GROUNDED — não rótulos arbitrários.

Honesto: é comunicação/sinalização emergente, NÃO linguagem humana plena (sem
gramática rica nem semântica composicional — isso é horizonte distante).
"""

from __future__ import annotations

import numpy as np


class SignalingGame:
    """Emissor (S: conceito->símbolo) + Receptor (R: símbolo->conceito) que
    aprendem um protocolo compartilhado por recompensa."""

    def __init__(self, n_concepts: int, n_symbols: int, lr: float = 0.25,
                 temp: float = 0.4, fail_penalty: float = 1.0,
                 init_noise: float = 0.05, seed: int = 0):
        self.nc = n_concepts
        self.ns = n_symbols
        self.lr = lr
        self.temp = temp
        self.fail_penalty = fail_penalty                  # fração de lr subtraída no erro
        self.rng = np.random.default_rng(seed)
        # ruído pequeno na init quebra empates/simetria (init zerada trava no concept 0)
        self.S = init_noise * self.rng.standard_normal((n_concepts, n_symbols))
        self.R = init_noise * self.rng.standard_normal((n_symbols, n_concepts))

    def _softmax(self, v):
        z = np.exp((v - v.max()) / self.temp)
        return z / z.sum()

    def play(self, concept: int, learn: bool = True, explore: bool = True):
        """Uma rodada. Retorna (símbolo, palpite, recompensa)."""
        # o EMISSOR explora símbolos (softmax); o RECEPTOR decodifica de forma
        # GULOSA (argmax) — recompensa menos ruidosa => convergência estável.
        if explore:
            m = int(self.rng.choice(self.ns, p=self._softmax(self.S[concept])))
        else:
            m = int(np.argmax(self.S[concept]))
        guess = int(np.argmax(self.R[m]))
        reward = int(guess == concept)
        if learn:
            # RECEPTOR: aprende o pareamento VERDADEIRO (atenção conjunta /
            # cross-situational — vê o referente), o que quebra o deadlock de
            # coordenação. EMISSOR: reforço modulado por recompensa (3º fator).
            self.R[m, concept] += self.lr
            self.S[concept, m] += self.lr if reward else -self.fail_penalty * self.lr
        return m, guess, reward

    def eval_accuracy(self) -> float:
        """Acurácia determinística (argmax) sobre todos os conceitos."""
        ok = 0
        for c in range(self.nc):
            m = int(np.argmax(self.S[c]))
            ok += int(int(np.argmax(self.R[m])) == c)
        return ok / self.nc

    def lexicon(self):
        """O código emergente: símbolo escolhido para cada conceito."""
        return [int(np.argmax(self.S[c])) for c in range(self.nc)]

    def is_bijection(self) -> bool:
        """O código é unívoco (cada conceito -> um símbolo distinto)?"""
        lex = self.lexicon()
        return len(set(lex)) == self.nc


class CompositionalGame:
    """M17 — comunicação de conceitos COMPOSTOS (atributo1 × atributo2) com
    mensagens de 2 símbolos. Testa se o código emergente GENERALIZA para
    combinações nunca vistas (composicionalidade = produtividade da linguagem).

    mode='compositional': símbolo por atributo (m1↔a1, m2↔a2) — pode generalizar.
    mode='holistic':      mensagem inteira por composto — memoriza, não generaliza.
    """

    def __init__(self, k1: int, k2: int, n_symbols: int, mode: str = "compositional",
                 lr: float = 0.15, temp: float = 0.4, init_noise: float = 0.05, seed: int = 0):
        self.k1, self.k2 = k1, k2
        self.nc = k1 * k2
        self.M = n_symbols
        self.mode = mode
        self.lr = lr
        self.temp = temp
        self.rng = np.random.default_rng(seed)
        rn = lambda *s: init_noise * self.rng.standard_normal(s)
        if mode == "compositional":
            self.S1 = rn(k1, n_symbols); self.S2 = rn(k2, n_symbols)   # atributo -> símbolo
            self.R1 = rn(n_symbols, k1); self.R2 = rn(n_symbols, k2)   # símbolo -> atributo
        else:                                                          # holístico
            self.Sf0 = rn(self.nc, n_symbols); self.Sf1 = rn(self.nc, n_symbols)
            self.Rf = rn(n_symbols * n_symbols, self.nc)               # par -> composto

    def _sm(self, v):
        z = np.exp((v - v.max()) / self.temp); return z / z.sum()

    def _pick(self, prefs, explore):
        return int(self.rng.choice(len(prefs), p=self._sm(prefs))) if explore else int(np.argmax(prefs))

    def play(self, a1: int, a2: int, learn: bool = True, explore: bool = True):
        if self.mode == "compositional":
            m1 = self._pick(self.S1[a1], explore); m2 = self._pick(self.S2[a2], explore)
            g1 = int(np.argmax(self.R1[m1])); g2 = int(np.argmax(self.R2[m2]))
            r1, r2 = int(g1 == a1), int(g2 == a2)
            if learn:
                self.R1[m1, a1] += self.lr; self.R2[m2, a2] += self.lr      # cross-situacional
                self.S1[a1, m1] += self.lr if r1 else -self.lr
                self.S2[a2, m2] += self.lr if r2 else -self.lr
            return (g1 == a1) and (g2 == a2)
        else:
            c = a1 * self.k2 + a2
            m1 = self._pick(self.Sf0[c], explore); m2 = self._pick(self.Sf1[c], explore)
            pair = m1 * self.M + m2
            g = int(np.argmax(self.Rf[pair])); ok = int(g == c)
            if learn:
                self.Rf[pair, c] += self.lr
                self.Sf0[c, m1] += self.lr if ok else -self.lr
                self.Sf1[c, m2] += self.lr if ok else -self.lr
            return bool(ok)

    def accuracy(self, composites):
        """Acurácia determinística num conjunto de compostos (a1,a2)."""
        return float(np.mean([self.play(a1, a2, learn=False, explore=False)
                              for (a1, a2) in composites]))


class GroundedLanguageGame:
    """M18 — comunicação ANCORADA na percepção (Harnad + language games de Steels).

    Os símbolos não referem ids dados, e sim CATEGORIAS PERCEPTUAIS: o emissor
    percebe um objeto (padrão), categoriza-o (com sua própria percepção), e nomeia;
    o receptor decodifica para sua categoria e identifica o objeto. O léxico gruda
    na percepção de cada um.

    Teste honesto (alinhamento): a percepção do RECEPTOR tem fidelidade controlável
    (`recv_noise`). Se ele não distingue dois objetos (os funde numa categoria), a
    comunicação sobre eles falha — só se comunica o que ambos conseguem distinguir.
    """

    def __init__(self, patterns, recv_noise: float = 0.0, n_symbols=None,
                 lr: float = 0.15, temp: float = 0.4, init_noise: float = 0.05, seed: int = 0):
        self.P = np.asarray(patterns)
        self.K = len(self.P)
        self.M = n_symbols or self.K
        self.lr = lr; self.temp = temp
        self.rng = np.random.default_rng(seed)
        # percepção do receptor: protótipos = objetos + ruído perceptual (fixos no run)
        protoR = self.P + recv_noise * self.rng.standard_normal(self.P.shape)
        protoR = protoR / (np.linalg.norm(protoR, axis=1, keepdims=True) + 1e-9)
        # a_R(i): categoria que o receptor percebe ao ver o objeto i
        self.aR = np.array([int(np.argmax(protoR @ self.P[i])) for i in range(self.K)])
        rn = lambda *s: init_noise * self.rng.standard_normal(s)
        self.S = rn(self.K, self.M)        # emissor: objeto(percebido) -> símbolo
        self.R = rn(self.M, self.K)        # receptor: símbolo -> categoria perceptual dele

    def _sm(self, v):
        z = np.exp((v - v.max()) / self.temp); return z / z.sum()

    def discriminability(self) -> float:
        """Fração de categorias perceptuais distintas que o receptor forma sobre os
        K objetos (1.0 = distingue todos; <1 = funde alguns)."""
        return len(set(self.aR.tolist())) / self.K

    def play(self, i: int, learn: bool = True, explore: bool = True):
        m = int(self.rng.choice(self.M, p=self._sm(self.S[i]))) if explore else int(np.argmax(self.S[i]))
        cr = int(np.argmax(self.R[m]))                 # categoria (do receptor) decodificada
        cands = [j for j in range(self.K) if self.aR[j] == cr]
        if cands:
            guess = int(self.rng.choice(cands))
        else:
            guess = int(self.rng.integers(self.K))
        reward = int(guess == i)
        if learn:
            self.R[m, self.aR[i]] += self.lr           # atenção conjunta: vê o alvo verdadeiro
            self.S[i, m] += self.lr if reward else -self.lr
        return reward

    def accuracy(self) -> float:
        """Acurácia esperada determinística (emissor/receptor gulosos)."""
        tot = 0.0
        for i in range(self.K):
            m = int(np.argmax(self.S[i]))
            cr = int(np.argmax(self.R[m]))
            cands = [j for j in range(self.K) if self.aR[j] == cr]
            if i in cands:
                tot += 1.0 / len(cands)                 # ambíguo se o receptor funde objetos
        return tot / self.K


class LivedLanguageGame:
    """M19 — a língua nasce da VIVÊNCIA. Diferente do M18, os conceitos não são
    dados: vêm da categorização que CADA agente aprendeu sozinho (um predictive
    coder por agente). Passa-se aqui o resultado desse aprendizado: arrays
    objeto -> conceito (a "leitura" que cada agente faz de cada objeto).

      sender_concepts[o] = conceito que o FALANTE formou para o objeto o
      receiver_concepts[o] = conceito que o OUVINTE formou para o objeto o

    O jogo então faz emergir um vocabulário sobre esses conceitos vividos. Se um
    agente aprendeu a confundir dois objetos (mesmo conceito), não há símbolo que
    os separe — a comunicação esbarra no que cada um aprendeu a distinguir.
    """

    def __init__(self, sender_concepts, receiver_concepts, n_symbols=None,
                 lr: float = 0.15, temp: float = 0.4, init_noise: float = 0.05, seed: int = 0):
        self.cs = np.asarray(sender_concepts, dtype=int)
        self.cr = np.asarray(receiver_concepts, dtype=int)
        self.K = len(self.cs)
        self.nCS = int(self.cs.max()) + 1
        self.nCR = int(self.cr.max()) + 1
        self.M = n_symbols or self.K
        self.lr = lr; self.temp = temp
        self.rng = np.random.default_rng(seed)
        rn = lambda *s: init_noise * self.rng.standard_normal(s)
        self.S = rn(self.nCS, self.M)        # conceito-do-falante -> símbolo
        self.R = rn(self.M, self.nCR)        # símbolo -> conceito-do-ouvinte

    def _sm(self, v):
        z = np.exp((v - v.max()) / self.temp); return z / z.sum()

    def sender_discrim(self):
        return len(set(self.cs.tolist())) / self.K

    def receiver_discrim(self):
        return len(set(self.cr.tolist())) / self.K

    def ceiling(self):
        """Teto teórico: fração de objetos que AMBOS os agentes distinguem (cada um
        com um conceito único para aquele objeto)."""
        def unique_mask(c):
            counts = {}
            for v in c: counts[v] = counts.get(v, 0) + 1
            return np.array([counts[v] == 1 for v in c])
        return float(np.mean(unique_mask(self.cs) & unique_mask(self.cr)))

    def play(self, o: int, learn: bool = True, explore: bool = True):
        a = self.cs[o]
        m = int(self.rng.choice(self.M, p=self._sm(self.S[a]))) if explore else int(np.argmax(self.S[a]))
        cr = int(np.argmax(self.R[m]))
        cands = [j for j in range(self.K) if self.cr[j] == cr]
        guess = int(self.rng.choice(cands)) if cands else int(self.rng.integers(self.K))
        reward = int(guess == o)
        if learn:
            self.R[m, self.cr[o]] += self.lr            # atenção conjunta
            self.S[a, m] += self.lr if reward else -self.lr
        return reward

    def accuracy(self):
        tot = 0.0
        for o in range(self.K):
            m = int(np.argmax(self.S[self.cs[o]]))
            cr = int(np.argmax(self.R[m]))
            cands = [j for j in range(self.K) if self.cr[j] == cr]
            if o in cands:
                tot += 1.0 / len(cands)
        return tot / self.K
