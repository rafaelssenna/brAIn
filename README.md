# brAIn

> Uma inteligência que **nasce sem saber nada** e aprende vivendo, no jeito do cérebro, não como um Transformer.

🌐 **Site/portfólio:** a jornada completa (36 marcos em 5 fases, com as figuras de cada experimento) vive no site,
em `index.html` + `site/`.

> Projeto de **Rafael Sena Roman**. Código aberto sob licença MIT. Ver [AUTHORS.md](AUTHORS.md).

## O que é isto

A IA dominante hoje (ChatGPT, Claude, Gemini) é um **Transformer**: aprende uma vez, num treino gigante e congelado, e depois nunca mais muda. O brAIn persegue o caminho oposto e biologicamente plausível:

- **Nasce ignorante.** Sem pré-treino, sem dataset da internet. Os "pesos" começam aleatórios.
- **Aprende online, vivendo.** Cada experiência muda a rede, em tempo real.
- **Sem backpropagation.** Usa regras de plasticidade *locais* (STDP / Hebbian) — como neurônios reais, que não têm como "propagar gradiente de volta".
- **É corporificado (embodied).** Tem sensores e atuadores num ambiente, fechando o laço percepção→ação→percepção.
- **Prevê o mundo.** A espinha dorsal é *predictive coding* / minimização de energia livre — a hipótese de que o cérebro é, antes de tudo, uma máquina de previsão.

## A honestidade que sustenta a ambição

Não vamos "espelhar 100% do cérebro humano" — **ninguém consegue isso hoje**, e os motivos são concretos (ver [docs/02-estado-da-arte.md](docs/02-estado-da-arte.md)):
- Não existe o mapa completo (o maior conectoma feito é o da mosca-da-fruta, ~140k neurônios, 2024).
- Não se conhece o algoritmo de aprendizado do cérebro.
- O custo computacional de simular fidelidade biológica é proibitivo.

O "100%" é a **estrela-guia**, não a entrega. A entrega é um sistema *genuinamente do paradigma cerebral* — aprende do zero, vivendo, com matemática real — em escala pequena mas crescente.

## Documentos

| Doc | Conteúdo |
|---|---|
| [01-visao-e-hipotese.md](docs/01-visao-e-hipotese.md) | O norte e a **hipótese científica falsificável** |
| [02-estado-da-arte.md](docs/02-estado-da-arte.md) | O que a ciência já fez (e não fez) — sem ilusão |
| [03-roadmap.md](docs/03-roadmap.md) | Os marcos, do primeiro neurônio vivo até o agente que se desenvolve |
| [04-fundacoes-matematicas.md](docs/04-fundacoes-matematicas.md) | A matemática: modelos de neurônio, STDP, predictive coding, energia livre |
| [05-prior-art.md](docs/05-prior-art.md) | **Alguém já fez isso?** Estado da arte com fontes — ninguém entregou a integração completa |
| [06-relatorio.md](docs/06-relatorio.md) | **Relatório científico completo** (capstone M1–M18): a jornada inteira, achados, recuos honestos, e até onde dá pra chegar |
| [07-fase2-integracao.md](docs/07-fase2-integracao.md) | **Fase 2 (M7→M13)**: costurar os módulos num organismo integrado — concluída |
| [08-fase3-cognicao.md](docs/08-fase3-cognicao.md) | **Fase 3 (M14→)**: cognição e linguagem emergentes — o "100%" distante (iniciada) |
| [09-estudo-grounding.md](docs/09-estudo-grounding.md) | **Estudo: symbol grounding** (prep. do M18) — Harnad, language games de Steels, com fontes |

## Interagir e treinar (não é chatbot — aprende vivendo)

O brAIn **nasce sem saber nada** e aprende vivendo. Duas ferramentas para isso:

```bash
python train.py            # treina o cérebro intensivamente (vive + consolida por replay/'sono')
                           #   -> salva em brain.npz; o aprendizado ACUMULA entre execuções
python chat.py             # converse: ensine palavras, pergunte, e veja-o aprender ao vivo
                           #   -> carrega brain.npz e ACORDA esperto (não renasce bebê)
```

No chat: `ensina <palavra>`, `ver <palavra>`, `pergunta <palavra>`, `treina tudo 40`, `status`,
`salvar`, `esquecer`. A persistência (`save`/`load` do `LivingAgent`) é a **memória de longo
prazo**; o replay no treino é a **consolidação** (o papel do sono). O `brain.npz` versionado já
vem com um cérebro que nomeia 100% das 6 palavras de português ancoradas na percepção.

## Status

**M0–M6 concluídos — roadmap completo** (30 testes verdes):
- **M1** neurônio LIF (valida vs teoria) · **M2** STDP / Pavlov (com controle) ·
  **M3** embodiment / fototaxia (98% vs 28%) · **M4** predictive coding (surpresa −79%,
  campos receptivos emergem, regra local sem backprop) · **M5** curiosidade
  (currículo ativo emerge; novelty colapsa no ruído) · **M6** ablação + escala +
  [relatório científico](docs/06-relatorio.md).
- **Fase 2 — COMPLETA** ([plano](docs/07-fase2-integracao.md)): **M7** organismo
  integrado (corpo+previsão+curiosidade) · **M8** memória/replay (vence o esquecimento) ·
  **M9** hierarquia (conceitos/invariância emergem) · **M10** predictive coding
  **spiking** (substrato da H1, −82% surpresa) · **M11** agência — planejar pela
  imaginação ("pensar", 100% vs 83%) · **M12** visão (detectores de borda emergem,
  como V1) · **M13** a costura final — tudo num agente só (só ele alcança o conteúdo
  atrás da parede: 10.7% vs 0%). **54 testes verdes.**
- **Fase 3 — em andamento** ([plano](docs/08-fase3-cognicao.md)): **M14** previsão
  temporal/sequencial (aprende sequências e **imagina a continuação**) · **M15**
  gramática (aprende REGRAS, gera frases novas 98% gramaticais) · **M16** comunicação
  **emergente** (dois agentes inventam uma língua, 100%) · **M17** proto-sintaxe —
  código **composicional** que generaliza (100% vs 0% holístico) · **M18** símbolos
  **ancorados na percepção** (Harnad; comunicação limitada pelo que se distingue,
  r=0.90) · **M19** a língua nasce da **vivência** (conceitos auto-aprendidos →
  língua 100%; comm acompanha o alinhamento, r=0.97) · **M20** o **organismo vivo** —
  num laço só, dois agentes percebem, lembram, são curiosos e falam ao mesmo tempo:
  percepção e linguagem **co-emergem** (surpresa 0.935→0.012, comunicação 8%→92%), com
  a linguagem ficando **atrás** da percepção · **M21** aprende **palavras de português**
  de verdade ancoradas no que percebe (nomear e apontar, 33%→100%; só nomeia o que
  distingue, r=0.99) · **M22** substrato **esparso** (método > força bruta): código
  cortical (12.5% ativo) faz o mesmo com **~7.5× menos operações sinápticas**, mantendo
  a qualidade · **M23** o **organismo vivo** (M20) roda nesse substrato esparso: percepção
  e linguagem co-emergem **gastando ~5.7× menos energia** (a fratura substrato↔cognição
  fechou) · **M24** o **organismo vivo percebe com neurônios que DISPARAM** — o substrato
  spiking (M10) entra dentro do organismo: percebendo por spikes reais, a surpresa cai
  0.93→0.068 e a comunicação co-emerge (75%), com a esparsidade (40% ativo) **emergindo
  sozinha** do limiar do disparo. A cognição **sobrevive aos spikes** · **M25** o
  **organismo CORPORIFICADO que fala** — costura as duas metades que viviam separadas:
  dois organismos com **corpo** navegam um mundo por curiosidade, percebem, lembram e
  **falam**, alinhando a língua só quando se **encontram** (atenção conjunta corporificada).
  A linguagem emerge mesmo quando ver custa andar (acaso 17% → ~75%), **gated pelos
  encontros**: perceber+mover+lembrar+ser curioso+falar, num laço só · **M26** o
  **organismo PLANEJA rotas num mundo 2D e fala** — corpo num grid com **parede**, aprende
  o modelo de transição vivendo, a curiosidade escolhe a meta, o **planejamento** traça a
  rota (contornando a parede), percebe, lembra e **nomeia** (atenção conjunta). Efeito de
  integração puro: **só quem planeja a rota alcança, percebe E nomeia o objeto atrás da
  parede** (nomeia C 100% vs 0% reativo; vê C com surpresa 0.003 vs 0.49) · **M27** a
  **primeira FRASE ancorada, aprendida ouvindo** — o agente percebe uma cena (forma×posição)
  dos pixels e aprende a montar/entender uma frase de 2 palavras só ouvindo, descobrindo o
  grounding E a ordem das palavras. O **mesmo cérebro aprende português E inglês** (ordens
  diferentes), produz/entende 100% e **generaliza** para frases nunca vistas; sem ordem fixa
  não aprende (controle). Falar começando, não decorar texto · **M28** **conversa emergente
  com frases** — dois agentes inventam, do zero e **sem professor**, uma língua com frases
  (forma×posição) só para se entenderem: o entendimento mútuo emerge (acaso 11% → 100%), o
  código **composicional generaliza** para cenas nunca ditas (75% vs 0% holístico), e a
  **ordem das palavras emerge consistente entre os dois** (uma gramática comum, arbitrária
  entre execuções como línguas humanas diferem). Do descrever ao conversar · **M29** **diálogo
  com turnos** — um agente PERGUNTA ("que forma?"/"que posição?"), o outro PERCEBE a cena e
  RESPONDE só a palavra certa. O diálogo emerge sem professor (17%→100%); o agente aprende a
  **responder à pergunta** (usar a pergunta pra escolher o que dizer): forma 100% e posição
  100%, enquanto um agente CEGO à pergunta acerta forma mas erra a posição (33%=acaso). Do
  conversar ao dialogar — o começo de raciocinar sobre a linguagem · **M30** **diálogo de
  vários turnos com contexto** — o objeto sobre o qual se falou fica em FOCO e o agente resolve
  a referência ("que forma?"→"barraH"; "e a posição DELA?"→"topo"). Com memória de contexto
  resolve "dela" 100%; sem contexto cai ao acaso (33%). Liga linguagem + memória — a conversa
  fica coerente no tempo · **M31** **correferência** — a cena tem VÁRIOS objetos e "dela" fica
  ambíguo; o agente resolve o referente certo pela **recência** (100% vs 51% ao acaso) ou por
  **descrição** ("a barraH, posição dela?" → escolhe pelo atributo, 100%). O começo de entender
  a ambiguidade da linguagem · **M32** **frases mais ricas** — descrição de **3 atributos**
  (forma+cor+posição, 27 cenas): "barra vermelho topo". O agente monta/entende 100% e
  **generaliza** para as cenas nunca vistas (100%), com o mesmo mecanismo aprendendo ordens de
  idiomas diferentes ("topo barra vermelho"); ordem aleatória não aprende (23%). A produtividade
  da linguagem — infinitos significados de poucas palavras · **M33** **recursão** — frases
  relacionais "estruturas dentro de estruturas": "barra vermelho **acima** ponto azul" (objeto-
  relação-objeto, 162 cenas). O agente compõe/entende 100% e **generaliza** para as relacionais
  nunca vistas (100%), reusando a sub-frase de objeto nas duas posições. O poder infinito da
  linguagem: poucas peças, infinitas frases · **M34** **negação** — "não o vermelho" é um
  operador lógico (= o complemento): o agente aponta o objeto pelo que ele **NÃO é** (100%),
  enquanto ignorar o "não" pega exatamente o errado (0%). Do nomear ao raciocinar com a
  linguagem · **M35** **descoberta não-supervisionada da gramática** — como um bebê ouvindo a
  língua: recebendo SÓ frases (sem rótulos), o agente descobre as **classes de palavras**
  (formas/cores/posições separam sozinhas, pureza 100%) e a **gramática** (ordem das classes) só
  pela estatística das co-ocorrências (Saffran 1996), distinguindo frases corretas de
  embaralhadas (100%/100%). De "alguém me ensina" para "eu descubro sozinho" · **M36**
  **quantificadores** — "todos/algum/nenhum são vermelhos?" é semântica de verdade: o agente
  checa a frase **contra a cena** (∀/∃/¬∃) e responde V/F (100% nos três), enquanto confundir os
  quantificadores erra "todos" e "nenhum" (geral 44%). Raciocinar com a linguagem, não só nomear.
  **151 testes verdes.**
  O "100%" (cognição/linguagem plenas) segue distante — cada marco é um tijolo real e verificado.

> ⚠️ **Honestidade:** o **M24** colocou o **spiking** (M10) dentro do organismo vivo, o
> **M25** deu **corpo** a ele (dois agentes negociando a língua por encontro), e o **M26**
> deu **planejamento num mundo 2D** com paredes, e o **M27** deu a **primeira FRASE** (sair
> de palavra solta para palavras com estrutura, em PT e EN), e o **M28** fez dois agentes
> **conversarem com frases sem professor** (língua e gramática emergentes), e o **M29** fez o
> **diálogo com turnos** (pergunta→resposta), o **M30** deu **memória de contexto** (resolve
> "dela"), o **M31** resolveu **correferência** entre vários objetos, e o **M32** subiu para
> **frases ricas de 3 atributos** que generalizam. Mas cada marco isola dimensões: as frases
> têm tamanho FIXO (não recursão, não tamanho variável, não fluência); a generalização emergente
> (M28) é parcial (75%); a correferência usa pistas simples; e ainda **não** existe um único
> agente com corpo 2D + planejamento + diálogo + frases ricas + substrato spiking/esparso, tudo
> junto — unir tudo e enriquecer (tamanho variável, recursão) é futuro (M33+). E
> raciocínio/linguagem plenos são a *estrela-guia emergente*
> ([docs/01](docs/01-visao-e-hipotese.md)), não entregas próximas. Tudo medido, verificado, e
> recuado onde os dados não sustentaram.
