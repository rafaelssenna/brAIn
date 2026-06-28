"""A primeira FRASE ancorada, aprendida ouvindo — M27 (rumo a falar).

Até aqui o brAIn grudava PALAVRAS soltas na percepção (M21: vê a barra no topo, diz
"topo"). Mas falar não é dizer palavras isoladas — é montar FRASES com estrutura. Este
marco dá o passo: o agente percebe uma cena de DOIS atributos (uma FORMA × uma POSIÇÃO)
e aprende, SÓ OUVINDO, a produzir e entender uma frase de duas palavras que a descreve.

O que ele descobre sozinho (ninguém programa as regras):
  1. GROUNDING — qual PALAVRA significa qual atributo (como M21, cross-situacional);
  2. SINTAXE  — qual POSIÇÃO na frase carrega a forma e qual carrega a posição
     (a ordem de palavras do idioma).

E o ponto central do foco "qualquer idioma": o MESMO mecanismo aprende ordens diferentes.
  • Português:  [forma] [posição]   — ex: "barra topo"   (modificador depois)
  • Inglês:     [posição] [forma]   — ex: "top bar"       (modificador antes)
Só muda o idioma de entrada (as frases ouvidas), não o código do agente. Ele aprende a
ordem de cada língua vivendo nela.

Honesto: escala de brinquedo (2 slots, poucos atributos, frase de tamanho fixo 2). É o
MECANISMO de aprender a primeira frase/sintaxe ancorada, como uma criança — NÃO fluência,
NÃO recursão, NÃO frases de tamanho variável (horizonte distante, provável híbrido).

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import numpy as np


class GroundedSentenceLearner:
    """Aprende frases ancoradas (forma × posição) só ouvindo, descobrindo grounding E ordem.

    A cena tem dois atributos independentes: forma ∈ {0..k_shape-1} e posição ∈
    {0..k_pos-1}. O vocabulário tem uma palavra por valor de forma e uma por valor de
    posição (palavras disjuntas). Uma frase é uma sequência de 2 palavras numa ORDEM fixa
    do idioma. O agente mantém:

      • para cada SLOT da frase (posição 0 e 1), um mapa palavra↔atributo APRENDIDO
        (não sabe de antemão se o slot 0 é forma ou posição);
      • um voto de qual PAPEL (forma/posição) cada slot carrega, estimado pela vivência.

    Ouvir uma frase com atenção conjunta (sabe a cena verdadeira): para cada slot, associa
    a palavra vista ali ao atributo correto e acumula evidência de qual papel o slot tem.
    """

    def __init__(self, k_shape: int, k_pos: int, lr: float = 0.2, seed: int = 0):
        self.k_shape = k_shape
        self.k_pos = k_pos
        self.n_slots = 2
        self.roles = ("shape", "pos")
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        # vocabulário: palavras de forma = [0..k_shape), palavras de posição = [k_shape..k_shape+k_pos)
        self.vocab = k_shape + k_pos
        # Para cada slot e cada PAPEL possível, um mapa atributo->palavra (produção) e
        # palavra->atributo (compreensão). O agente não sabe qual papel é o slot; aprende.
        rn = lambda *s: 0.01 * self.rng.standard_normal(s)
        # produção: dado (slot, papel, valor-do-atributo) -> preferência por palavra
        self.prod = {
            ("shape"): rn(self.n_slots, k_shape, self.vocab),   # [slot, forma]  -> palavra
            ("pos"): rn(self.n_slots, k_pos, self.vocab),       # [slot, posição]-> palavra
        }
        # compreensão: dado (slot, palavra) -> evidência de cada valor de cada papel
        self.comp_shape = rn(self.n_slots, self.vocab, k_shape)  # [slot, palavra]->forma
        self.comp_pos = rn(self.n_slots, self.vocab, k_pos)      # [slot, palavra]->posição
        # voto de papel por slot: quanto cada slot "parece" carregar forma vs posição
        self.role_evidence = np.zeros((self.n_slots, 2))         # col 0=shape, 1=pos

    # ---------- ouvir (aprender uma frase, com atenção conjunta à cena) ----------
    def hear(self, sentence, shape: int, pos: int):
        """Ouve uma frase (lista de 2 palavras) sabendo a cena (forma, posição) — atenção
        conjunta. Aprende grounding por slot e acumula evidência de papel por slot."""
        truth = {"shape": shape, "pos": pos}
        for slot, word in enumerate(sentence):
            # grounding: a palavra deste slot indica a forma E a posição verdadeiras
            # (a evidência se concentra no papel certo porque a palavra de forma só
            #  aparece junto da forma, e idem posição — cross-situacional).
            self.comp_shape[slot, word, shape] += self.lr
            self.comp_pos[slot, word, pos] += self.lr
            self.prod["shape"][slot, shape, word] += self.lr
            self.prod["pos"][slot, pos, word] += self.lr
            # evidência de papel: se a palavra está no intervalo de forma, este slot
            # tende a carregar forma; idem posição (descoberto, não dado).
            if word < self.k_shape:
                self.role_evidence[slot, 0] += 1.0
            else:
                self.role_evidence[slot, 1] += 1.0

    def slot_role(self, slot: int) -> str:
        """O papel que o agente APRENDEU para um slot (forma ou posição)."""
        return "shape" if self.role_evidence[slot, 0] >= self.role_evidence[slot, 1] else "pos"

    def learned_order(self):
        """A ordem de palavras aprendida: papel de cada slot. Ex: ('shape','pos')=PT."""
        return tuple(self.slot_role(s) for s in range(self.n_slots))

    # ---------- produzir (vê a cena, fala a frase na ordem aprendida) ----------
    def say(self, shape: int, pos: int):
        """Vê a cena e produz a frase de 2 palavras, cada slot no papel que aprendeu."""
        out = []
        for slot in range(self.n_slots):
            role = self.slot_role(slot)
            val = shape if role == "shape" else pos
            out.append(int(np.argmax(self.prod[role][slot, val])))
        return out

    # ---------- entender (ouve a frase, identifica a cena) ----------
    def understand(self, sentence):
        """Ouve uma frase e identifica (forma, posição) usando o papel de cada slot."""
        shape_guess = pos_guess = None
        for slot, word in enumerate(sentence):
            role = self.slot_role(slot)
            if role == "shape":
                shape_guess = int(np.argmax(self.comp_shape[slot, word]))
            else:
                pos_guess = int(np.argmax(self.comp_pos[slot, word]))
        # fallback se algum papel não apareceu (slots degenerados)
        if shape_guess is None:
            shape_guess = int(np.argmax(self.comp_shape[0, sentence[0]]))
        if pos_guess is None:
            pos_guess = int(np.argmax(self.comp_pos[1, sentence[-1]]))
        return shape_guess, pos_guess


# ---------- idiomas: a ordem das palavras (o que muda entre línguas) ----------

def make_language(order, k_shape: int):
    """Devolve uma função que, dada (forma, posição), monta a frase do idioma na ORDEM dada.

    `order` é uma tupla de papéis por slot, ex: ('shape','pos') (PT) ou ('pos','shape') (EN).
    Palavras: forma -> id [0..k_shape); posição -> id [k_shape..). Palavras disjuntas, então
    o agente pode descobrir o papel de cada slot pela própria palavra que ouve.
    """
    def to_sentence(shape: int, pos: int):
        word_of = {"shape": shape, "pos": k_shape + pos}
        return [word_of[role] for role in order]
    return to_sentence


PT_ORDER = ("shape", "pos")     # "barra topo"  — substantivo + modificador
EN_ORDER = ("pos", "shape")     # "top bar"     — modificador + substantivo
