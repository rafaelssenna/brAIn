"""Frases mais ricas: descrição de N atributos — M32 (a produtividade da linguagem).

O M27 montou frases de 2 atributos (forma + posição). Uma descrição de verdade é mais rica:
"a barra VERMELHA no TOPO" — forma, COR e posição. Este marco generaliza o aprendiz de
frases para N ATRIBUTOS (aqui 3): o agente percebe os N atributos de uma cena e monta/entende
uma frase de N palavras, ainda descobrindo SÓ OUVINDO o grounding (palavra↔atributo) e a ORDEM
das palavras (qual slot carrega qual papel).

Com 3 atributos × 3 valores há 27 cenas possíveis. Treinando numa fração e testando nas NUNCA
vistas, prova-se a PRODUTIVIDADE da linguagem: montar descrições novas, que nunca se ouviu —
"infinitos significados de poucas palavras". É o salto que mais mostra composicionalidade real.

E, como no M27, o MESMO mecanismo aprende ORDENS diferentes (idiomas com a ordem dos modificadores
trocada), só vivendo na língua.

Honesto: escala de brinquedo (3 atributos, 3 valores, frase de tamanho fixo N). É o MECANISMO
da descrição composicional rica, NÃO fluência, NÃO recursão, NÃO tamanho variável — horizonte
distante, provável híbrido.

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np


class RichSentenceLearner:
    """Aprende frases de N atributos só ouvindo, descobrindo grounding E ordem dos slots.

    `k_values` é a lista [k_0, k_1, ...] com o nº de valores de cada atributo (papel). As
    palavras são DISJUNTAS por atributo (cada atributo tem seu bloco de palavras), então o
    agente pode descobrir o papel de cada slot pela palavra que ouve. O papel de cada slot é
    decidido por EXCLUSIVIDADE MÚTUA: cada slot fica com um papel distinto (uma divisão de
    trabalho), então a frase nunca colapsa dois slots no mesmo papel.
    """

    def __init__(self, k_values, lr: float = 0.2, seed: int = 0):
        self.k = list(k_values)
        self.n_attr = len(self.k)
        self.n_slots = self.n_attr
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        # vocabulário: blocos disjuntos, um por atributo. offset[a] = início do bloco do attr a.
        self.offset = np.cumsum([0] + self.k[:-1]).tolist()
        self.vocab = sum(self.k)
        rn = lambda *s: 0.01 * self.rng.standard_normal(s)
        # produção: por (slot, atributo, valor) -> preferência por palavra
        self.prod = [rn(self.n_slots, self.k[a], self.vocab) for a in range(self.n_attr)]
        # compreensão: por (slot, palavra) -> evidência de cada valor de cada atributo
        self.comp = [rn(self.n_slots, self.vocab, self.k[a]) for a in range(self.n_attr)]
        # evidência de qual papel (atributo) cada slot carrega
        self.role_ev = np.zeros((self.n_slots, self.n_attr))

    # ---------- ouvir (atenção conjunta: conhece os valores verdadeiros) ----------
    def hear(self, sentence, values):
        """Ouve uma frase (N palavras) sabendo os valores (uma tupla de N atributos)."""
        for slot, word in enumerate(sentence):
            for a in range(self.n_attr):
                self.comp[a][slot, word, values[a]] += self.lr
                self.prod[a][slot, values[a], word] += self.lr
            # qual atributo a palavra deste slot indica (pelo bloco de vocabulário a que pertence)
            attr = self._word_attr(word)
            if attr is not None:
                self.role_ev[slot, attr] += 1.0

    def _word_attr(self, word):
        for a in range(self.n_attr):
            if self.offset[a] <= word < self.offset[a] + self.k[a]:
                return a
        return None

    # ---------- papéis dos slots por EXCLUSIVIDADE MÚTUA (atribuição gulosa) ----------
    def slot_roles(self):
        """Atribui a cada slot um atributo distinto, maximizando a evidência total (guloso).
        Garante uma divisão de trabalho: nenhum atributo fica em dois slots."""
        ev = self.role_ev.copy()
        roles = [None] * self.n_slots
        used = set()
        # ordena pares (slot, attr) por evidência decrescente e atribui sem repetir
        pairs = sorted(((ev[s, a], s, a) for s in range(self.n_slots) for a in range(self.n_attr)),
                       reverse=True)
        for _, s, a in pairs:
            if roles[s] is None and a not in used:
                roles[s] = a; used.add(a)
        # preenche qualquer slot restante com um atributo livre
        for s in range(self.n_slots):
            if roles[s] is None:
                free = next(a for a in range(self.n_attr) if a not in used)
                roles[s] = free; used.add(free)
        return roles

    def learned_order(self):
        """A ordem aprendida: o atributo (papel) de cada slot."""
        return tuple(self.slot_roles())

    # ---------- produzir / entender ----------
    def say(self, values):
        """Vê os N atributos e produz a frase de N palavras, cada slot no papel aprendido."""
        roles = self.slot_roles()
        return [int(np.argmax(self.prod[roles[s]][s, values[roles[s]]])) for s in range(self.n_slots)]

    def understand(self, sentence):
        """Ouve a frase e devolve a tupla de N atributos (valor de cada papel)."""
        roles = self.slot_roles()
        out = [0] * self.n_attr
        for slot, word in enumerate(sentence):
            a = roles[slot]
            out[a] = int(np.argmax(self.comp[a][slot, word]))
        return tuple(out)


# ---------- idiomas: a ordem dos slots (o que muda entre línguas) ----------

def make_rich_language(order, k_values):
    """Devolve uma função values->frase que monta a frase na ORDEM dada (tupla de atributos
    por slot). Palavras disjuntas por atributo (bloco com offset)."""
    offset = np.cumsum([0] + list(k_values[:-1])).tolist()

    def to_sentence(values):
        return [offset[a] + values[a] for a in order]
    return to_sentence
