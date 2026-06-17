# brAIn

> Uma inteligência que **nasce sem saber nada** e aprende vivendo, no jeito do cérebro, não como um Transformer.

🌐 **Site/portfólio:** a jornada completa (18 marcos, com as figuras de cada experimento) vive no site,
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
  a qualidade. **84 testes verdes.** O "100%"
  (cognição/linguagem plenas) segue distante — cada marco é um tijolo real e verificado.

> ⚠️ **Honestidade:** o M13 costura corpo+previsão+curiosidade+memória+
> planejamento+hierarquia num agente só (rate-based). Mas o substrato **spiking**
> (M10) ainda **não** roda dentro desse agente integrado ao mesmo tempo — uni-los é
> trabalho futuro. E raciocínio/linguagem são a *estrela-guia emergente*
> ([docs/01](docs/01-visao-e-hipotese.md)), não entregas próximas. Tudo medido,
> verificado, e recuado onde os dados não sustentaram.
