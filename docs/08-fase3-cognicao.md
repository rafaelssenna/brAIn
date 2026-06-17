# 08 — Fase 3: Cognição e Linguagem Emergentes (o "100%")

A estrela-guia do projeto: raciocínio e linguagem **aprendidos, não programados**,
emergindo da mesma máquina de previsão. É a inteligência geral pela via biológica —
distante, incerta, AGI-hard. Esta fase avança em direção a ela, com a mesma
disciplina (mede tudo, baseline, ablação, verificação, recuar onde não sustenta) e
**honestidade brutal sobre a distância**.

## O caminho (hipótese de rota, não promessa)

```
previsão temporal (M14) → estrutura sequencial/gramática → conceitos compostos
   → símbolos ancorados (symbol grounding) → COMUNICAÇÃO emergente
```

Cada degrau é a previsão sequencial (M14) ficando mais rica e mais abstrata.
A linguagem, no fim, é predição sequencial sobre conceitos ancorados na experiência
— por isso a previsão temporal é a fundação certa. (Para linguagem plena, um
**híbrido** com módulo de linguagem pode ser o realista — decisão futura.)

## Marcos

### M14 — Previsão temporal/sequencial ✅ (concluído)
**Objetivo:** prever SEQUÊNCIAS no tempo, não só o instante — o substrato de
imaginar cadeias de futuros e a ponte para a linguagem.
- [x] `src/brain/temporal.py`: `TemporalPredictiveCoder` — contexto = traço com
      vazamento do passado; previsão `x̂ = W·contexto`; aprendizado LOCAL
      `ΔW ∝ erro·contextoᵀ` (sem backprop).
- [x] 4 testes — `tests/test_temporal.py` (total: **58 verdes**)

**Resultados (`experiments/m14_temporal.py`):**
- ✅ **Aprende estrutura temporal**: erro final **0.024** numa sequência estruturada
      vs **0.895** numa aleatória (controle — não inventa estrutura onde não há).
- ✅ **Imagina a continuação**: semeado com 0,1,2, gera `3,4,0,1,2,3,4,0` —
      exatamente a continuação real. "Pensar à frente no tempo" funciona.
- ✅ **O contexto importa**: numa sequência em que o próximo depende do passado,
      COM memória erra menos (0.219) que SEM memória / 1ª ordem (0.265) — efeito
      modesto mas real (o traço com vazamento desambigua parcialmente).

**Significado:** o brAIn deixou de prever só o agora e passou a prever o **futuro
sequencial** — e a imaginá-lo. É o primeiro tijolo da cognição: a base sobre a qual
gramática, composição e (muito adiante) linguagem podem ser construídas.
**Pergunta respondida:** *sim — o brAIn aprende e imagina sequências no tempo.*

### M15 — Gramática: aprender REGRAS sequenciais ✅ (concluído)
**Objetivo:** sair de "decorar uma sequência" para aprender uma GRAMÁTICA (autômato
com transições válidas e probabilidades) e GERAR sequências novas válidas.
- [x] `experiments/m15_grammar.py`: lê do preditor temporal (M14) uma DISTRIBUIÇÃO
      sobre o próximo símbolo (softmax das similaridades com os protótipos).
- [x] 2 testes — `tests/test_grammar.py` (total: **60 verdes**)

**Resultados:**
- ✅ **Aprende a matriz de transição**: a gramática aprendida bate com a real
      (erro médio **0.022**), incluindo os PONTOS DE RAMIFICAÇÃO (0→{1,2}, 3→{4,5})
      e suas probabilidades.
- ✅ **Gera frases novas válidas**: sequências geradas são **98% gramaticais** vs
      **22%** do acaso. Honesto: ~2% de transições escapam (readout/amostragem
      imperfeitos), não 100%.

**Significado:** o brAIn deixou de copiar uma sequência e passou a internalizar suas
REGRAS — e a produzir sequências novas que as respeitam. É o ensaio direto da
SINTAXE: a estrutura sobre a qual a linguagem se apoia.

### M16 — Comunicação emergente ✅ (concluído)
**Objetivo:** dois agentes inventarem, do zero, um código compartilhado
símbolo↔conceito — o primeiro "conversar".
- [x] `src/brain/communication.py`: `SignalingGame` (jogo de referência). Emissor
      explora símbolos; receptor decodifica; aprendizado modulado por recompensa
      (3 fatores) + **atenção conjunta** (o receptor aprende o pareamento verdadeiro,
      como na aquisição de palavras — o que quebra o deadlock de coordenação).
- [x] 3 testes — `tests/test_communication.py` (total: **63 verdes**)

**Resultados (`experiments/m16_communication.py`):**
- ✅ **A comunicação EMERGE**: acurácia sobe do acaso (20%) a **100%**; em 5/5
      sementes converge para comunicação perfeita (não é sorte).
- ✅ **Código unívoco**: o léxico emergente é uma bijeção (cada conceito → um
      símbolo distinto) — uma matriz de permutação limpa.
- ✅ Os agentes de fato se entendem: todas as trocas de exemplo corretas.
- 🔧 Risco materializou e foi resolvido honestamente: a regra só-recompensa
      travava (deadlock de coordenação); a atenção conjunta (cross-situacional)
      resolveu — convergência robusta.

**Significado:** dois brAIn inventaram um vocabulário SOZINHOS, ancorado em
conceitos, pela necessidade de coordenar. É o primeiro tijolo real do "conversar".
**Honesto:** é sinalização emergente grounded, **não linguagem plena** (sem
gramática rica nem semântica composicional). Unir isto à gramática (M15) e escalar
para linguagem de verdade é o horizonte — provavelmente híbrido.

### M17 — Proto-sintaxe / composicionalidade ✅ (concluído)
**Objetivo:** de "palavras isoladas" (M16) para "frases com estrutura" — comunicar
conceitos COMPOSTOS (atributo1 × atributo2) com mensagens de 2 símbolos, e testar se
o código GENERALIZA para combinações nunca vistas (a produtividade da linguagem).
- [x] `src/brain/communication.py`: `CompositionalGame` (modos composicional vs
      holístico). Treina em 6 de 9 combinações; testa nas 3 NUNCA vistas.
- [x] 3 testes — `tests/test_compositional.py` (total: **66 verdes**)

**Resultados (`experiments/m17_compositional.py`) — média de 6 seeds:**
- ✅ **Composicional GENERALIZA**: acerta **100%** das combinações nunca vistas —
      os símbolos viraram ATRIBUTOS, então combinações novas se decodificam sozinhas.
      Essa é a produtividade da sintaxe: infinitos significados de poucos símbolos.
- ✅ **Holístico só decora**: 100% no treino, **0%** no nunca-visto — sem estrutura,
      não há generalização. O contraste mostra que a ESTRUTURA é o que dá produtividade.

**Honesto:** damos aos agentes a CAPACIDADE de compor (codificação por atributo); o
que EMERGE é o código consistente e a generalização. Linguagem plena (recursão,
semântica profunda, mensagens de tamanho variável) segue no horizonte — provável híbrido.

### M18 — Comunicação ancorada na percepção ✅ (concluído)
**Objetivo:** os símbolos referirem CATEGORIAS PERCEPTUAIS (Harnad), não ids dados —
o primeiro grounding real. Precedido de estudo da literatura ([09](09-estudo-grounding.md):
Harnad, language games de Steels, cross-situational learning).
- [x] `src/brain/communication.py`: `GroundedLanguageGame` — emissor percebe→
      categoriza→nomeia; receptor decodifica→identifica o objeto; percepção do
      receptor com fidelidade controlável (teste de alinhamento).
- [x] 3 testes — `tests/test_grounded.py` (total: **69 verdes**)

**Resultados (`experiments/m18_grounded.py`):**
- ✅ **Comunicação ancorada funciona** (percepções alinhadas): acaso (16%) → **100%**,
      com um léxico bijetivo onde cada objeto PERCEBIDO tem seu símbolo.
- ✅ **Achado honesto (alinhamento)**: a comunicação é **limitada pela
      discriminabilidade perceptual** — correlação **r = 0.90** entre o quanto o
      receptor distingue os objetos e a acurácia de comunicação; os pontos não
      ultrapassam o limite perceptual (y=x). *Só se comunica o que ambos distinguem.*

**Significado:** pela primeira vez os símbolos significam algo que o agente
APRENDEU A PERCEBER — não etiquetas dadas. E o grounding revela uma verdade real:
linguagem exige conceitos compatíveis; a percepção limita o que se pode dizer.
**Honesto:** escala de brinquedo; não linguagem plena.

### Próximos (horizonte distante)
- Recursão / mensagens de tamanho variável; ancorar a comunicação no organismo
  integrado (M13) com conceitos do M9 aprendidos online; semântica mais rica; e,
  para linguagem plena, provável híbrido. O "100%" segue sendo o norte.

> Honestidade permanente: estamos longe do "100%". Cada marco é um tijolo real e
> verificado; a casa inteira (cognição/linguagem plenas) é o horizonte, não a
> entrega de amanhã.
