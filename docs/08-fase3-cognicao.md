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

### M19 — A língua nasce da vivência ✅ (concluído)
**Objetivo:** fechar o laço percepção→linguagem. Os conceitos não são dados: cada
agente APRENDE os seus, sozinho, com um predictive coder (M4/M9), e só então os
dois inventam uma língua sobre esses conceitos vividos.
- [x] `src/brain/communication.py`: `LivedLanguageGame` — joga sobre os mapas
      objeto→conceito que cada agente aprendeu (não etiquetas dadas).
- [x] 3 testes — `tests/test_lived.py` (total: **72 verdes**)

**Resultados (`experiments/m19_lived_language.py`) — H7 confirmada:**
- ✅ **A língua emerge de conceitos AUTO-APRENDIDOS**: quando os dois agentes
      aprendem a distinguir os objetos, a comunicação vai do acaso (12%) a **100%**.
      Pela 1ª vez os símbolos significam conceitos que o próprio agente formou pela
      experiência, não rótulos que demos.
- ✅ **Achado honesto (alinhamento)**: a comunicação acompanha o quanto o agente
      mais fraco aprendeu a distinguir — **r = 0.97** entre a discriminabilidade de
      conceitos de B e a acurácia, sempre sob o teto (y=x). Só se fala do que ambos
      aprenderam a separar.

**Significado:** é o laço completo — percepção (M4/M9, conceitos auto-aprendidos) →
linguagem (M16/M18) — em miniatura. O mais perto do norte: cognição e comunicação
nascendo juntas da vivência.
**Honesto:** escala de brinquedo; o alinhamento dos conceitos depende de ambos
aprenderem a separar as mesmas coisas (experiência compartilhada ajuda, mas o jogo
de linguagem é que termina de alinhar).

### M20 — O organismo vivo: a costura da cognição ✅ (concluído)
**Objetivo:** num ÚNICO laço online, dois agentes ao mesmo tempo **percebem** e
formam conceitos (M4/M9), **lembram** por replay (M8), são **curiosos** (learning
progress, M5) e aprendem a **falar** (M16/M19) sobre o que percebem. A diferença
para o M19: lá os conceitos eram aprendidos ANTES, offline; aqui percepção e
linguagem nascem JUNTAS, vivendo. **Pergunta científica:** dá para ancorar a
linguagem em conceitos que ainda estão se formando?
- [x] `src/brain/living.py`: `LivingAgent` — junta predictive coder (percepção) +
      replay buffer (memória) + matrizes símbolo↔conceito (linguagem) num agente só;
      `learn_perception` devolve a surpresa para alimentar a curiosidade.
- [x] 2 testes — `tests/test_living.py` (total: **74 verdes**)

**Resultados (`experiments/m20_living_mind.py`):**
- ✅ **Percepção e linguagem CO-EMERGEM no mesmo laço**: vivendo, a surpresa cai
      **0.935 → 0.012** (aprendeu a ver) e a comunicação sobe **8% → 92%** (aprendeu
      a falar) — sem etapas separadas, tudo online.
- ✅ **A linguagem fica ATRÁS da percepção**: o domínio perceptual (surpresa domada)
      assenta por volta do passo ~1200; a comunicação continua subindo bem depois
      disso, só fechando perto do fim. Primeiro aprender a ver, depois aprender a
      falar — exatamente a ordem esperada.
- ✅ **Conceitos formados vivendo**: os campos receptivos do agente A (barras
      orientadas) emergem do laço, não de treino prévio.

**Honesto:** escala de brinquedo (6 objetos, D=64). Conceitos em formação são um
**alvo móvel**: enquanto a percepção não assenta, a linguagem tem que perseguir
representações que ainda mudam — por isso ela custa mais a travar. A medida ingênua
de "objetos distinguidos" já nasce alta (códigos aleatórios separam por acaso); o
sinal honesto de aprender a ver é a **surpresa caindo**, e é nela que a figura se apoia.

**Significado:** é o organismo inteiro num laço só — perceber, lembrar, ter
curiosidade e falar, todos da mesma vida. Fecha a Fase 3 com a costura da cognição:
o mais perto que chegamos do norte em miniatura.

### M21 — Palavras de PORTUGUÊS ancoradas na percepção ✅ (concluído)
**Objetivo:** sair dos símbolos inventados e aprender PALAVRAS REAIS do português, do
jeito de uma criança: o agente vê o mundo (barras em posições), forma seus conceitos
sozinho (M4) e um "professor" diz a palavra certa junto (atenção conjunta). Aprende as
duas direções da língua — NOMEAR (vê e diz a palavra) e APONTAR (ouve e aponta).
- [x] `experiments/m21_portuguese.py` (reusa o `LivingAgent` do M20): vocabulário real
      `topo, meio, base, esquerda, centro, direita` ancorado em conceitos auto-aprendidos.
- [x] 3 testes — `tests/test_portuguese.py` (total: **77 verdes**)

**Resultados:**
- ✅ **Aprende as palavras vivendo**: nomear **33% → 100%** e compreender **31% → 100%**.
      No fim ele vê `topo` e diz "topo", ouve "direita" e aponta o objeto certo — nas
      seis palavras.
- ✅ **A língua vem ATRÁS da percepção**: nomear/apontar só assentam depois que os
      conceitos estabilizam (mesma assinatura do M20).
- ✅ **Achado honesto (o teto perceptual)**: nomear só vai até onde ele DISTINGUE as
      coisas — correlação **r = 0.99** entre discriminabilidade perceptual e acurácia ao
      nomear, sempre sob o teto (y=x). Percepção pobre → ele confunde objetos e erra os
      nomes. *Só se nomeia o que se percebe.*

**Significado:** é o MECANISMO de aprender palavras como uma criança (grounding
cross-situacional), agora com palavras de português de verdade. A língua significa o
que o agente APRENDEU A VER, não rótulos abstratos.
**Honesto:** 6 palavras, mundo de brinquedo. É o mecanismo, não fluência; português
pleno pede outra escala (ou um híbrido).

### M22 — Substrato esparso e dirigido a eventos ✅ (concluído)
**Objetivo:** perseguir o cérebro pelo caminho CERTO, sem força bruta. O cérebro roda
com <20W porque é ESPARSO (poucos neurônios ativos por vez) e dirigido a eventos (só
quem dispara custa). Este marco mede esse princípio sobre a máquina de previsão (M4):
um código latente esparso (k-winners-take-all) atinge a mesma qualidade gastando muito
menos OPERAÇÕES SINÁPTICAS (SynOps). A moeda é a CONTAGEM de operações (independente de
hardware, o "AC op" da literatura neuromórfica), NÃO o relógio.
- [x] `src/brain/sparse_predictive.py`: `SparsePredictiveCoder` (herda do M4; k-WTA e/ou
      L1 via soft-threshold; aprendizado local ΔW∝ε·rᵀ só nas colunas ativas) + `OpCounter`
      (conta MACs denso vs dirigido a eventos) + fórmulas dos baselines densos.
- [x] 7 testes — `tests/test_sparse_predictive.py` (total: **84 verdes**); inclui o teste
      de degeneração (k=None, l1=0 ⇒ idêntico ao M4).

**Resultados (`experiments/m22_sparse_efficient.py`):**
- ✅ **Esparsidade cortical de verdade**: o `nonneg`/ReLU que já existia fica em **74%
      ativo** (quase denso!); o k-WTA (k=2) traz para **12.5%** — faixa cortical. Ou seja,
      o k-WTA **não é redundante** com o ReLU: a ablação decisiva passou.
- ✅ **Qualidade preservada**: surpresa held-out **0.203 (denso) vs 0.238 (esparso)** —
      levemente pior, perto do ótimo de subespaço, com **100% de discriminação** de
      conceitos nos dois.
- ✅ **Energia**: **~7.5× menos SynOps por inferência** que o denso pleno; e, honrando o
      M6, **~5.5× menos SynOps ATÉ atingir a qualidade-alvo** (não é truque de "por passo").
- ✅ **Fronteira de Pareto (onde QUEBRA)**: varrendo k, a discriminação fica 100% de k=2
      a 16; em **k=1 (6.2% ativo) ela COLAPSA para 75%** — esparsidade agressiva demais
      funde conceitos. O ponto-doce é k=2 (cortical, discriminação plena, energia mínima).

**Honesto:** a moeda é OPERAÇÃO SINÁPTICA, não relógio — em Python/NumPy o laço esparso
é até mais lento (o M6 já mostrou). Isto prova o **princípio de eficiência em miniatura**
(esparsidade mantém a qualidade cortando operações); NÃO prova escala, nem wall-clock, nem
eficiência biológica (estamos a ~9 ordens de magnitude do cérebro). O termo bottom-up Wᵀε
é denso por natureza; o ganho limpo vem da previsão e do update dirigidos a eventos.

**Significado:** o primeiro passo do programa "rumo ao cérebro pelo método, não pela força
bruta". Próximos: rodar o organismo (M20/M21) sobre o substrato esparso (M23); event-driven
de verdade + a fronteira honesta de hardware neuromórfico (M24).

### Próximos (horizonte distante)
- M23: organismo (M20/M21) sobre o substrato esparso. M24: event-driven real + onde a
  CPU/Python para e o neuromórfico começa. Recursão / mensagens de tamanho variável;
  semântica mais rica; e, para linguagem plena, provável híbrido. O "100%" segue sendo o norte.

> Honestidade permanente: estamos longe do "100%". Cada marco é um tijolo real e
> verificado; a casa inteira (cognição/linguagem plenas) é o horizonte, não a
> entrega de amanhã.
