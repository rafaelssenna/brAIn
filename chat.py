"""Chat com o brAIn — ensine a IA ao vivo e veja ela aprender (não é chatbot).

Diferente de um chatbot (que decorou texto e nunca mais muda), aqui você CONVERSA com um
organismo que NASCE SEM SABER NADA e APRENDE VIVENDO, em tempo real, ancorando palavras no
que percebe. Você mostra cenas, ensina palavras (como um pai apontando objetos para um bebê),
pergunta, e vê a rede mudar a cada interação.

O mundinho: barras em 6 posições. As palavras de português descrevem ONDE a barra está:
  topo · meio · base · esquerda · centro · direita

Comandos (digite e tecle Enter):
  ver <palavra>      mostra a cena daquela posição e o agente tenta NOMEAR o que vê
  ensina <palavra>   mostra a cena E diz a palavra certa junto (você ensina; ele aprende)
  treina <palavra> <n>   ensina aquela palavra n vezes seguidas (prática)
  treina tudo <n>    pratica TODAS as palavras n vezes cada (uma "aula")
  pergunta <palavra> ele OUVE a palavra e tenta APONTAR a posição certa
  status             mostra o quanto ele já aprendeu (nomear, distinguir)
  palavras           lista as palavras que ele conhece
  ajuda              mostra estes comandos
  sair               encerra

Uso:   python chat.py

Projeto de Rafael Sena Roman. Ver AUTHORS.md.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from brain import LivingAgent  # noqa: E402

np.seterr(all="ignore")

SIDE = 8
D = SIDE * SIDE
NOISE = 0.06
ROWS = [1, 4, 6]
COLS = [1, 4, 6]
WORDS = ["topo", "meio", "base", "esquerda", "centro", "direita"]
K = len(WORDS)


def objects():
    pats = []
    for r in ROWS:
        p = np.zeros((SIDE, SIDE)); p[r, :] = 1.0; pats.append(p.ravel())
    for c in COLS:
        p = np.zeros((SIDE, SIDE)); p[:, c] = 1.0; pats.append(p.ravel())
    P = np.array(pats)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


P = objects()
WORD_IDX = {w: i for i, w in enumerate(WORDS)}


def draw_scene(i):
    """Desenha a cena (barra) em ASCII, para você VER o que o agente vê."""
    g = np.zeros((SIDE, SIDE))
    if i < 3:
        g[ROWS[i], :] = 1.0
    else:
        g[:, COLS[i - 3]] = 1.0
    return "\n".join("  " + "".join("█" if v else "·" for v in row) for row in g)


def noisy(i, rng):
    v = P[i] + NOISE * rng.standard_normal(D)
    return v / np.linalg.norm(v)


class Chat:
    def __init__(self, seed=1):
        self.agent = LivingAgent(D, n_latent=12, n_symbols=K, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.taught = 0

    # --- ações ---
    def ver(self, word):
        i = WORD_IDX[word]
        obj = noisy(i, self.rng)
        self.agent.learn_perception(obj)                 # vê (aprende a perceber, sem rótulo)
        said = self.agent.speak(self.agent.concept(obj), explore=False)
        print(draw_scene(i))
        mark = "OK" if said == i else "ainda não"
        print(f'  [mostrei "{word}"]  ele olha e diz: "{WORDS[said]}"   ({mark})')

    def ensina(self, word):
        i = WORD_IDX[word]
        obj = noisy(i, self.rng)
        c = self.agent.concept(obj)
        self.agent.learn_perception(obj)                 # vê
        self.agent.learn_listen(i, c)                    # ouve a palavra certa (atenção conjunta)
        self.agent.reinforce_speak(c, i, True)           # associa seu conceito à palavra
        self.taught += 1
        said = self.agent.speak(self.agent.concept(obj), explore=False)
        print(f'  ensinei "{word}". agora, ao ver, ele diz: "{WORDS[said]}"'
              f'   ({"aprendeu!" if said == i else "ainda formando"})')

    def treina(self, word, n):
        for _ in range(n):
            i = WORD_IDX[word]
            obj = noisy(i, self.rng)
            c = self.agent.concept(obj)
            self.agent.learn_perception(obj)
            self.agent.learn_listen(i, c)
            self.agent.reinforce_speak(c, i, True)
            self.taught += 1
        print(f'  pratiquei "{word}" {n}x. ', end="")
        self._report_word(word)

    def treina_tudo(self, n):
        order = list(range(K))
        for _ in range(n):
            self.rng.shuffle(order)
            for i in order:
                obj = noisy(i, self.rng)
                c = self.agent.concept(obj)
                self.agent.learn_perception(obj)
                self.agent.learn_listen(i, c)
                self.agent.reinforce_speak(c, i, True)
                self.taught += 1
        print(f'  aula completa: {n}x cada palavra ({n*K} exemplos).')
        self.status()

    def pergunta(self, word):
        i = WORD_IDX[word]
        # ele ouve a palavra e aponta um conceito; mostramos a cena que melhor casa
        guess_concept = self.agent.listen(i)
        # qual objeto ele percebe como esse conceito?
        concepts = [self.agent.concept(P[j]) for j in range(K)]
        pointed = [j for j in range(K) if concepts[j] == guess_concept]
        if len(pointed) == 1 and pointed[0] == i:
            print(f'  ouço "{word}" -> ele aponta a posição certa:')
            print(draw_scene(i))
        elif pointed:
            print(f'  ouço "{word}" -> ele aponta "{WORDS[pointed[0]]}"'
                  f'   ({"certo" if pointed[0] == i else "ainda confunde"})')
        else:
            print(f'  ouço "{word}" -> ele ainda não sabe apontar (precisa aprender mais)')

    def _report_word(self, word):
        i = WORD_IDX[word]
        said = self.agent.speak(self.agent.concept(P[i]), explore=False)
        print(f'ao ver "{word}", ele diz "{WORDS[said]}" ({"certo" if said == i else "errado"})')

    def status(self):
        nomear = np.mean([self.agent.speak(self.agent.concept(P[i]), explore=False) == i
                          for i in range(K)])
        disc = self.agent.discriminability([P[i] for i in range(K)])
        print(f'  --- o que ele já sabe (após {self.taught} exemplos ensinados) ---')
        print(f'  nomear certo:    {100*nomear:.0f}%  das {K} palavras')
        print(f'  distingue:       {100*disc:.0f}%  das posições (percepção)')
        # mini-tabela
        for i in range(K):
            said = self.agent.speak(self.agent.concept(P[i]), explore=False)
            mark = "OK " if said == i else " x "
            print(f'    vê "{WORDS[i]:9s}" -> diz "{WORDS[said]:9s}" {mark}')

    def palavras(self):
        print("  palavras que ele pode aprender:  " + " · ".join(WORDS))


HELP = __doc__.split("Comandos", 1)[1].split("Uso:", 1)[0]
HELP = "Comandos" + HELP


def main():
    print("=" * 64)
    print("  brAIn — chat: ensine a IA e veja ela aprender (não é chatbot)")
    print("=" * 64)
    print("  Ele NASCE SEM SABER NADA. Ensine com 'ensina <palavra>' ou")
    print("  'treina tudo 30', e veja-o aprender. Digite 'ajuda' para os comandos.")
    print(HELP)
    chat = Chat()
    while True:
        try:
            line = input("\nvocê> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  até logo!"); break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        try:
            if cmd in ("sair", "exit", "quit"):
                print("  até logo!"); break
            elif cmd in ("ajuda", "help"):
                print(HELP)
            elif cmd == "palavras":
                chat.palavras()
            elif cmd == "status":
                chat.status()
            elif cmd == "ver" and len(parts) == 2 and parts[1] in WORD_IDX:
                chat.ver(parts[1])
            elif cmd == "ensina" and len(parts) == 2 and parts[1] in WORD_IDX:
                chat.ensina(parts[1])
            elif cmd == "pergunta" and len(parts) == 2 and parts[1] in WORD_IDX:
                chat.pergunta(parts[1])
            elif cmd == "treina" and len(parts) == 3 and parts[1] == "tudo":
                chat.treina_tudo(int(parts[2]))
            elif cmd == "treina" and len(parts) == 3 and parts[1] in WORD_IDX:
                chat.treina(parts[1], int(parts[2]))
            else:
                print("  não entendi. digite 'ajuda' para ver os comandos"
                      " (e use só as palavras conhecidas: 'palavras').")
        except Exception as e:  # noqa: BLE001 — REPL robusto a entradas estranhas
            print(f"  ops, algo deu errado: {e}")


if __name__ == "__main__":
    main()
