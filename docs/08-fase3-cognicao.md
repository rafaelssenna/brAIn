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

### M23 — O organismo vivo no substrato eficiente ✅ (concluído)
**Objetivo:** costurar a fratura entre o substrato eficiente (M22) e o organismo
integrado (M20). O MESMO organismo do M20 (dois agentes que percebem, lembram e falam)
roda sobre o código esparso, lado a lado com o denso. A cognição (a co-emergência
percepção↔linguagem) sobrevive à esparsidade? E quanto de energia se economiza?
- [x] `src/brain/living.py`: `LivingAgent` ganha `sparse_k`/`l1` (injeta o
      `SparsePredictiveCoder` do M22; com `sparse_k=None` é idêntico ao M20/M21, sem regressão).
- [x] `experiments/m23_efficient_organism.py`: organismo denso vs esparso, mesma vida/semente.
- [x] 2 testes — `tests/test_efficient_organism.py` (total: **86 verdes**).

**Resultados:**
- ✅ **A cognição SOBREVIVE à esparsidade**: percepção e linguagem co-emergem no
      substrato esparso — comunicação **67% (denso) e 83% (esparso)**, conceitos **100%**
      distinguidos nos dois.
- ✅ **Atividade cortical**: o organismo denso fica **89.6% ativo**; o esparso, **12.5%**.
- ✅ **Energia**: **~5.7× menos SynOps por percepção** no organismo esparso.

**Honesto:** o resultado é de uma semente; a comunicação mais alta no esparso (83 vs 67)
está dentro do ruído entre execuções — a afirmação é que a cognição **sobrevive** e
co-emerge com muito menos energia, NÃO que o esparso seja superior em linguagem. Moeda =
operação sináptica, não relógio; escala de brinquedo; princípio, não eficiência cerebral.

**Significado:** a fratura "substrato eficiente ↔ organismo vivo" fechou — o organismo
inteiro (perceber + lembrar + falar) roda no código esparso. Próximo: event-driven de
verdade (M24) e a fronteira honesta do que a CPU faz vs o que pede hardware neuromórfico.

### M24 — O organismo vivo PERCEBENDO com neurônios que disparam ✅ (concluído)
**Objetivo:** fechar a maior lacuna honesta do projeto (marcada com ⚠️ no README e no
M13): o organismo vivo (M20, dois agentes que percebem, lembram e falam) sempre percebeu
com um substrato RATE-BASED (taxa contínua). Mas o cérebro não faz isso — neurônios
DISPARAM (spikes). O M10 já tinha esse substrato spiking (neurônios LIF reais), mas vivia
ISOLADO, fora do organismo. Aqui ele entra no laço da vida. **Pergunta científica:** a
cognição (percepção + linguagem co-emergindo) sobrevive quando a percepção é feita por
neurônios que de fato disparam?
- [x] `src/brain/spiking_predictive.py`: `SpikingPerceptionCoder` — adaptador do M10 que é
      DROP-IN no organismo (mesma API do M4/M22: `infer(x)->r`, `learn`, `prediction_error`,
      `active_fraction`, contagem de SynOps via `OpCounter`). Não muda a matemática do
      substrato spiking; só ajusta a assinatura, como o `SparsePredictiveCoder` (M22) fez.
- [x] `src/brain/living.py`: `LivingAgent` ganha o flag `spiking=True` (injeta o coder
      spiking; com `spiking=False` é idêntico ao M20/M21/M23 — sem regressão).
- [x] 5 testes — `tests/test_spiking_organism.py` (total: **91 verdes**).

**Resultados (`experiments/m24_spiking_organism.py`) — rate-based vs spiking, mesma vida/semente:**
- ✅ **A cognição SOBREVIVE aos spikes**: percebendo com neurônios LIF que disparam, a
      surpresa cai **0.93 → 0.068** (vs **0.006** rate-based) e a comunicação co-emerge
      (**75%** spiking vs **67%** rate), com **100%** dos conceitos distinguidos nos dois.
- ✅ **Esparsidade EMERGENTE (de graça)**: o organismo rate-based fica **89.6% ativo**; o
      spiking, **39.6%** — e ninguém impôs k-WTA. A esparsidade nasce do LIMIAR do spike
      (corrente sublimiar => o neurônio não dispara => ρ=0). É a retificação biológica
      surgindo sozinha, não programada.
- ✅ **O substrato é spiking de verdade**: o raster (painel c) mostra os neurônios LIF
      disparando ao perceber — eventos discretos no tempo, não números contínuos.

**Honesto:** escala de brinquedo (6 objetos, D=64); o laço spiking é LENTO em Python
(~80ms/passo) — a moeda continua sendo OPERAÇÃO SINÁPTICA, não o relógio (M6/M22). O
spiking é *rate-coded* (a TAXA de disparo carrega o valor), não código temporal puro — o
spiking predictive coding com timing fino segue como trabalho futuro. A surpresa do spiking
assenta um pouco acima do rate-based (substrato mais cru). Isto prova que a cognição roda
sobre spikes reais EM MINIATURA; não prova escala nem eficiência biológica.

**Significado:** é o passo mais "cérebro de verdade" do organismo até aqui — a percepção
que sustenta a linguagem deixou de ser uma taxa abstrata e passou a EMERGIR de neurônios
que disparam. A fratura spiking↔cognição (aberta desde o M10/M13) fechou em miniatura.

### M25 — O organismo CORPORIFICADO que fala ✅ (concluído)
**Objetivo:** costurar as duas metades do cérebro que viviam separadas. Até aqui havia
um agente COM CORPO (M7/M13: navega, percebe, é curioso, lembra) que era MUDO, e um
agente que FALA (M20→M24: percebe com spikes, lembra, fala) que não tinha CORPO (os
objetos chegavam até ele). Um cérebro humano é UM organismo só. Aqui dois organismos
COMPLETOS vivem num anel de lugares: cada um tem corpo, navega por CURIOSIDADE (M5),
percebe o objeto de onde está (M4/M24), lembra (M8) e fala (M16/M21) — e só alinham a
língua quando se ENCONTRAM no mesmo lugar (atenção conjunta corporificada).
- [x] `src/brain/embodied_language.py`: `EmbodiedLanguageAgent` — monta o `LivingAgent`
      (M24, percepção+memória+linguagem) num CORPO que vive no `RingWorld` (M7), com a
      `IntrinsicMotivation` (M5) guiando a navegação. Aceita `spiking=True`/`sparse_k`.
- [x] 5 testes — `tests/test_embodied_language.py` (total: **96 verdes**).

**Resultados (`experiments/m25_embodied_language.py`):**
- ✅ **O organismo inteiro vive num laço só**: percebe (surpresa **0.90 → 0.010**),
      distingue **100%** dos objetos, navega o mundo por curiosidade e fala.
- ✅ **A linguagem EMERGE mesmo quando ver custa andar** (H8): a comunicação sobe do
      acaso (17%) a **~75%**, comparável ao baseline SEM corpo (objetos mostrados, ~58%
      nesta semente) — a corporificação não impede a língua de emergir.
- ✅ **Gated pela atenção conjunta** (o sinal mais limpo, painel c): a comunicação
      ACOMPANHA o número de ENCONTROS co-localizados. Só se nomeia o que os dois corpos
      atenderam JUNTOS — a mesma verdade do M18/M19/M21 (só se fala do que ambos
      percebem), agora imposta pelo CORPO: ver e alinhar custam ir até lá.

**Honesto:** corpo e sem-corpo ficam COMPARÁVEIS — a diferença varia com a semente, então
NÃO afirmamos que um vence o outro (só que a língua sobrevive à corporificação). Escala de
brinquedo; "encontro" é co-localização simples (não percepção mútua rica); é o MECANISMO
do organismo inteiro num laço, não um cérebro pronto.

**Significado:** é a metade que faltava. Pela primeira vez um organismo do brAIn faz, no
MESMO laço, tudo o que antes vivia em agentes separados: **perceber + mover + lembrar +
ser curioso + falar**. O mais perto de um "organismo cerebral" inteiro que o projeto chegou.

### M26 — O organismo que PLANEJA rotas num mundo 2D e fala ✅ (concluído)
**Objetivo:** o M25 deu corpo ao organismo que fala, mas num anel trivial (sem obstáculos).
Um cérebro de verdade PLANEJA: imagina o caminho, contorna paredes, alcança o que quer.
Esse maquinário já existia (M11/M13: `GridWorld` com paredes + `WorldModel` aprendido +
`plan()` por busca na imaginação), mas vivia no agente MUDO. Aqui ele se junta à linguagem.
**Pergunta científica (H9):** o planejamento permite ao organismo PERCEBER e NOMEAR o que
está atrás da parede — algo que o reativo (sem imaginar o desvio) nunca alcança?
- [x] `src/brain/planning_language.py`: `PlanningLanguageAgent` — `LivingAgent`
      (percepção/memória/linguagem, M24/M25) montado num corpo 2D (`GridWorld`) + `WorldModel`/
      `plan` (M11) + curiosidade (M5). A língua é aprendida por ATENÇÃO CONJUNTA (M21): ao
      chegar numa célula de objeto, um "professor" diz a palavra e o agente aprende a nomear/
      apontar. Aceita `spiking`/`sparse_k`. `use_planning=False` cai no reativo (trava na parede).
- [x] 5 testes — `tests/test_planning_language.py` (total: **101 verdes**).

**Resultados (`experiments/m26_planning_language.py`) — média de 3 sementes, planejador vs reativo:**
- ✅ **Só quem planeja ALCANÇA o objeto atrás da parede**: visitas a C **302× vs 0×**.
      O reativo anda reto e trava na parede; nunca chega a C.
- ✅ **Só quem planeja APRENDE A VÊ-LO**: surpresa em C **0.003 (planejador) vs 0.494
      (reativo)** — o reativo nunca vê C de perto, então nunca aprende a percebê-lo.
- ✅ **Só quem planeja APRENDE A NOMEÁ-LO**: nomeia C **100% vs 0%**; nomeia o conjunto
      todo **100% (planejador) vs 67% (reativo)**. O reativo nomeia os 2 objetos acessíveis,
      mas nunca o terceiro (atrás da parede), porque nunca chegou lá.
- ✅ **Aprende o modelo de transição vivendo**: nasce sem saber o que suas ações fazem e
      preenche (s,a)→s' ao agir, base para imaginar a rota.

**Significado:** é o efeito de INTEGRAÇÃO em estado puro — **pensar a rota é o que permite
perceber E nomear o que está escondido**. Perceber + planejar + mover + lembrar + curiosidade
+ nomear, tudo num laço 2D só; nenhuma peça sozinha faz isso. O organismo cerebral mais
completo do projeto até aqui.

**Honesto:** escala de brinquedo (grid 5×5, 3 objetos); a língua é aprendida por PROFESSOR
(atenção conjunta do ambiente, mecanismo do M21), NÃO negociada entre dois agentes (isso foi
o M25) — dois corpos curiosos raramente se encontram na mesma célula num grid com parede,
então o professor é o caminho honesto para isolar o efeito do planejamento sobre a língua. É
o MECANISMO do organismo, não um cérebro pronto.

### M27 — A primeira FRASE ancorada, aprendida ouvindo (PT e EN) ✅ (concluído)
**Objetivo:** o brAIn grudava PALAVRAS soltas na percepção (M21). Mas falar é montar
FRASES com estrutura. Este marco dá o passo: o agente PERCEBE uma cena de dois atributos
(uma FORMA × uma POSIÇÃO, em pixels), forma seus conceitos sozinho (M4) e aprende, SÓ
OUVINDO, a produzir e entender uma frase de duas palavras que a descreve — descobrindo o
GROUNDING (palavra↔atributo) E a SINTAXE (qual posição da frase carrega qual papel).
**O ponto central do foco "qualquer idioma":** o MESMO mecanismo aprende ordens diferentes
(PT `[forma][posição]` = "barra topo"; EN `[posição][forma]` = "top bar") só vivendo na língua.
**Pergunta científica (H10):** dá para aprender a primeira frase com estrutura — grounding +
ordem de palavras — só ouvindo, e o mesmo cérebro serve para idiomas de ordens diferentes?
- [x] `src/brain/grounded_sentence.py`: `GroundedSentenceLearner` — aprende, por slot da
      frase, o mapa palavra↔atributo e qual PAPEL (forma/posição) o slot carrega (descoberto,
      não dado). `make_language(order)` monta frases de um idioma; `PT_ORDER`/`EN_ORDER`.
- [x] 5 testes — `tests/test_grounded_sentence.py` (total: **106 verdes**).

**Resultados (`experiments/m27_grounded_sentence.py`):**
- ✅ **Percepção REAL**: o agente forma seus conceitos de forma e posição dos PIXELS
      (distingue formas 100%, posições 100%) — a frase gruda no que ele percebe, não em ids.
- ✅ **O MESMO mecanismo aprende PT E EN**: produzir/entender **100% nos dois**, com a ORDEM
      aprendida diferente em cada (`(forma,posição)` no PT, `(posição,forma)` no EN). Vê a
      barra horizontal no topo → PT "barraH topo", EN "top barraH".
- ✅ **GENERALIZA (produtividade)**: treina em 6 de 9 combinações, monta frases certas para
      as 3 **nunca vistas** — **100%**. Os símbolos viraram atributos componíveis (como M17).
- ✅ **Baseline honesto (estrutura real)**: se o idioma EMBARALHA a ordem a cada frase (sem
      regra fixa), entender despenca (**~64%** vs 100%) — prova que aprende SINTAXE, não decora.

**Significado:** é a primeira FRASE — sair de palavra solta para palavras com ESTRUTURA,
ancoradas no que percebe, aprendida ouvindo. E o mais perto do norte "qualquer idioma": o
mesmo cérebro aprende a ordem de línguas diferentes só vivendo nelas. É falar começando, não
decorar texto: o oposto do chatbot (que congela e não ancora).

**Honesto:** escala de brinquedo — frase de 2 palavras, tamanho fixo, poucos atributos. É o
MECANISMO da sintaxe ancorada (como uma criança junta as 2 primeiras palavras), NÃO fluência,
NÃO recursão, NÃO frases de tamanho variável (horizonte distante, provável híbrido). As
palavras de forma/posição são disjuntas, então o slot dá pista do papel — realista (o
aprendiz usa a palavra para inferir a estrutura), mas é o caso mais simples da sintaxe.

### M28 — Conversa emergente com FRASES (do descrever ao conversar) ✅ (concluído)
**Objetivo:** o M27 aprendeu a primeira frase OUVINDO um professor (que já sabia a resposta).
Conversar é dois cérebros se entendendo SEM professor — coordenando do zero, como dois bebês
inventando uma língua. Este marco dá o passo: dois agentes inventam, juntos, uma mini-língua
com FRASES de duas palavras (forma × posição) só para se entenderem sobre cenas.
**Pergunta científica (H11):** quando dois cérebros PRECISAM se coordenar, emerge uma língua
com FRASES — vocabulário E uma ORDEM de palavras consistente entre eles? E o código
composicional GENERALIZA para cenas que nunca disseram um ao outro?
- [x] `src/brain/emergent_sentence.py`: `EmergentSentenceGame` — dois agentes simétricos
      (falam e ouvem); jogo de referência com frases de 2 slots; sem professor (reforço por
      sucesso + atenção conjunta, M16). O PAPEL de cada slot (forma/posição) emerge por
      EXCLUSIVIDADE MÚTUA (os slots competem pelos papéis), então a frase nunca colapsa.
      `mode='compositional'` (símbolo por atributo) vs `'holistic'` (decora a frase inteira).
- [x] 5 testes — `tests/test_emergent_sentence.py` (total: **111 verdes**).

**Resultados (`experiments/m28_emergent_sentence.py`) — média de 8 sementes:**
- ✅ **A conversa EMERGE sem professor**: entendimento mútuo do acaso (11%) a **100%** —
      dois agentes se coordenam sozinhos sobre uma língua de frases.
- ✅ **Composicional GENERALIZA; holístico decora**: nas cenas NUNCA ditas, composicional
      **75%** vs holístico **0%** (treino 100% em ambos). A estrutura é o que dá produtividade.
- ✅ **A ORDEM das palavras EMERGE consistente entre os dois agentes**: em **100%** das
      execuções os dois convergem para a MESMA ordem (uma gramática comum). E a ordem é
      ARBITRÁRIA entre execuções (5/8 sementes "forma+posição", 3/8 "posição+forma") —
      como línguas humanas diferem, mas cada conversa é internamente coerente.

**Significado:** é o passo de "descrever" para "conversar". Pela primeira vez no projeto, dois
cérebros inventam uma língua com FRASES (não só palavras, M16) — com vocabulário E gramática
(ordem) — só pela necessidade de se entender, sem ninguém ensinar. Composicional, então
produtiva. É conversar começando, não decorar texto.

**Honesto:** escala de brinquedo (frase de 2 palavras, tamanho fixo, poucos atributos); a
generalização sem professor é parcial (75%, não 100% — coordenar é mais difícil que ouvir um
professor). É o MECANISMO da conversa composicional emergente, NÃO fluência, NÃO recursão,
NÃO tamanho variável — horizonte distante, provável híbrido.

### Próximos (horizonte distante)
- M29: frases mais ricas (3+ atributos, tamanho variável, talvez recursão simples) e diálogo
  com TURNOS (pergunta/resposta); unir TUDO num agente só — corpo 2D + planejamento + conversa
  + substrato spiking/esparso. Semântica mais profunda; para linguagem plena, provável híbrido.
  O "100%" segue sendo o norte.

> Honestidade permanente: estamos longe do "100%". Cada marco é um tijolo real e
> verificado; a casa inteira (cognição/linguagem plenas) é o horizonte, não a
> entrega de amanhã.
