"""Descobrir a gramática por ESTATÍSTICA pura — M35 (a aquisição não-supervisionada).

Até aqui um "professor" dava atenção conjunta: o agente sabia o que cada palavra significava.
Aqui é mais duro e mais parecido com um bebê ouvindo a língua ao redor: o agente recebe SÓ um
corpus de FRASES (sequências de palavras), sem NENHUM rótulo — nem do significado, nem da
estrutura — e tem que DESCOBRIR sozinho:
  1. as CLASSES de palavras (quais palavras se comportam igual: cores juntas, formas juntas);
  2. a GRAMÁTICA (quais sequências de classes são válidas).

Tudo pela ESTATÍSTICA das co-ocorrências (probabilidade de transição entre palavras), como na
aprendizagem estatística de linguagem em bebês (Saffran, Aslin & Newport, 1996). Sem backprop,
sem rótulos — distribuição de contextos + agrupamento.

A intuição: palavras da mesma CLASSE aparecem nos MESMOS contextos (as cores aparecem todas
depois de uma forma, antes de uma posição, etc.). Então agrupar palavras por similaridade de
contexto recupera as classes; e a matriz de transição entre classes é a gramática.

Pergunta científica (H18): o agente descobre as classes de palavras e a gramática de um corpus
SÓ pela estatística — sem rótulos — e usa isso para distinguir frases gramaticais de
embaralhadas?

Honesto: vocabulário pequeno, gramática regular (sem recursão), escala de brinquedo. É o
MECANISMO da descoberta estatística de classes/gramática, NÃO indução de gramática plena.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np


class GrammarDiscovery:
    """Descobre classes de palavras e transições válidas de um corpus de frases, sem rótulos.

    Conta bigramas (palavra seguinte) e, para cada palavra, monta um VETOR DE CONTEXTO
    (com quem ela co-ocorre: antes e depois). Palavras com contextos parecidos são agrupadas
    (k-means simples sobre os vetores de contexto) — essas são as CLASSES. A matriz de
    transição entre classes é a GRAMÁTICA aprendida.
    """

    def __init__(self, vocab_size: int, n_classes: int, seed: int = 0):
        self.V = vocab_size
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)
        self.before = np.zeros((vocab_size, vocab_size))   # before[a,b] = b veio antes de a
        self.after = np.zeros((vocab_size, vocab_size))    # after[a,b]  = b veio depois de a
        self.labels = None                                 # classe descoberta de cada palavra
        self.trans = None                                  # matriz de transição entre classes

    # ---------- 1) ouvir o corpus (só contar co-ocorrências) ----------
    def observe(self, sentence):
        for i, w in enumerate(sentence):
            if i > 0:
                self.before[w, sentence[i - 1]] += 1
            if i < len(sentence) - 1:
                self.after[w, sentence[i + 1]] += 1

    def observe_corpus(self, sentences):
        for s in sentences:
            self.observe(s)

    # ---------- 2) descobrir as CLASSES (agrupar por similaridade de contexto) ----------
    def _context_vectors(self):
        # contexto de cada palavra = [com quem vem antes | com quem vem depois], normalizado
        ctx = np.hstack([self._norm_rows(self.before), self._norm_rows(self.after)])
        return ctx

    @staticmethod
    def _norm_rows(m):
        s = m.sum(axis=1, keepdims=True)
        return m / (s + 1e-9)

    def discover_classes(self, n_iter=50):
        """k-means simples (cosseno) sobre os vetores de contexto -> classes de palavras."""
        X = self._context_vectors()
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        # init: escolhe n_classes palavras distantes como centróides
        idx = [int(self.rng.integers(self.V))]
        for _ in range(self.n_classes - 1):
            d = 1 - Xn @ Xn[idx].T               # distância cosseno aos já escolhidos
            idx.append(int(np.argmax(d.min(axis=1))))
        C = Xn[idx].copy()
        labels = np.zeros(self.V, dtype=int)
        for _ in range(n_iter):
            sim = Xn @ C.T
            labels = np.argmax(sim, axis=1)
            for k in range(self.n_classes):
                members = Xn[labels == k]
                if len(members):
                    v = members.mean(axis=0)
                    C[k] = v / (np.linalg.norm(v) + 1e-9)
        self.labels = labels
        return labels

    # ---------- 3) a GRAMÁTICA: transições entre classes ----------
    def discover_grammar(self):
        if self.labels is None:
            self.discover_classes()
        T = np.zeros((self.n_classes, self.n_classes))
        for w in range(self.V):
            for nxt in range(self.V):
                if self.after[w, nxt] > 0:
                    T[self.labels[w], self.labels[nxt]] += self.after[w, nxt]
        self.trans = self._norm_rows(T)
        return self.trans

    # ---------- usar a gramática: uma frase é gramatical? ----------
    def sentence_score(self, sentence):
        """Pontuação da frase = a MENOR probabilidade de transição de classe ao longo dela.
        Usa o mínimo (não a média) porque UMA transição proibida já torna a frase agramatical
        — basta um elo quebrado para a cadeia não valer."""
        if self.trans is None:
            self.discover_grammar()
        if len(sentence) < 2:
            return 1.0
        ps = []
        for i in range(len(sentence) - 1):
            a, b = self.labels[sentence[i]], self.labels[sentence[i + 1]]
            ps.append(self.trans[a, b])
        return float(np.min(ps))

    def is_grammatical(self, sentence, thresh=0.05):
        """Gramatical se NENHUMA transição de classe da frase for (quase) proibida."""
        return self.sentence_score(sentence) >= thresh

    # ---------- avaliação da descoberta de classes (contra um gabarito) ----------
    def class_purity(self, true_classes):
        """Pureza: fração de palavras cuja classe descoberta concorda com a maioria da sua
        classe verdadeira (métrica padrão de clustering, invariante a permutação de rótulos)."""
        if self.labels is None:
            self.discover_classes()
        total = 0
        for k in range(self.n_classes):
            members = np.where(self.labels == k)[0]
            if len(members) == 0:
                continue
            tc = [true_classes[w] for w in members]
            total += max(tc.count(x) for x in set(tc))
        return total / self.V
