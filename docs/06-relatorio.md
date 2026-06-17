# 06 — Relatório Científico (capstone completo: M1–M18)

Consolidação honesta da jornada inteira do brAIn — do primeiro neurônio à
comunicação ancorada. Cada afirmação aqui passou por testes automatizados
(**69 verdes**) e/ou verificação adversarial independente. O princípio que
sustentou tudo: **mecanismo biologicamente plausível, medir tudo, baseline sempre,
ablação, e recuar onde os dados não sustentam.** Ciência, não mágica.

## Resumo

Construímos, do zero e em NumPy, um cérebro que aprende vivendo — fora do paradigma
Transformer/backprop — e o levamos do neurônio único até dois agentes inventando
uma língua ancorada na percepção. **18 marcos, 3 fases, 69 testes verdes.** O
resultado é a primeira demonstração end-to-end (em miniatura, honesta) da
integração que a literatura nomeou mas ninguém entregou ([05](05-prior-art.md)):
perceber → prever → agir → ser curioso → lembrar → conceituar → planejar → ver →
comunicar com estrutura e grounding. Não é o cérebro humano; é o *paradigma certo*,
montado tijolo por tijolo, com um mapa sincero do que falta.

## Fase 1 — Os ingredientes (M1–M6)

| Marco | Entrega | Achado validado |
|---|---|---|
| **M1** | Neurônio LIF | Simulação bate com a curva f–I analítica |
| **M2** | Sinapse STDP | Condicionamento pavloviano; controle não-pareado não aprende |
| **M3** | Embodiment | Fototaxia 98% vs 28%; laço percepção→ação→percepção fecha |
| **M4** | Predictive coding | Surpresa −79% (held-out); campos receptivos emergem; regra LOCAL |
| **M5** | Curiosidade | Currículo ativo emerge; novelty colapsa na "TV chiando" (98% preso) |
| **M6** | Ablação + escala | Aprendizado/inferência/capacidade são essenciais; vetorização ingênua não escala |

**Recuos honestos da Fase 1** (a verificação adversarial forçou): o "piso do ruído"
do M4 estava na fórmula errada (corrigido p/ ótimo de subespaço); os "estágios" do
M5 vêm da dificuldade intrínseca, não da curiosidade; a curiosidade **não** supera
o aleatório num mundo fácil; vetorizar dá throughput mas perde em eficiência amostral.

## Fase 2 — O organismo integrado (M7–M13)

| Marco | Entrega | Achado |
|---|---|---|
| **M7** | Laço integrado | Corpo+previsão+curiosidade num agente; navega, evita ruído 4× |
| **M8** | Memória / replay | Esquecimento +0.44 → +0.004; **destrava a curiosidade** (93%→99%) |
| **M9** | Hierarquia | Invariância de categoria emerge (dentro-da-categoria 0.81→0.99) |
| **M10** | Substrato spiking | Predictive coding em neurônios LIF; surpresa −82%; a fratura M4 costurada |
| **M11** | Agência / planejar | Imagina futuros e planeja: 100% vs 83% reativo; "pensar à frente" |
| **M12** | Visão | Detectores de borda orientados emergem, como V1 |
| **M13** | Costura final | Tudo num laço; só ele alcança conteúdo atrás da parede (10.7% vs 0%) |

A Fase 2 **resolveu lacunas da Fase 1**: o M8 curou o esquecimento que derrubava a
curiosidade; o M9 deu profundidade (conceitos) à camada única; o M10 costurou a
fratura rate-vs-spiking; o M13 integrou tudo. *Recuo honesto:* no M13, o domínio
por-lugar é ruidoso (dinâmica estocástica) — medimos o efeito robusto (chegar a C).

## Fase 3 — A cognição (M14–M18)

| Marco | Entrega | Achado |
|---|---|---|
| **M14** | Previsão temporal | Aprende sequências e **imagina a continuação** correta |
| **M15** | Gramática | Aprende REGRAS; gera frases novas 98% gramaticais (vs 22%) |
| **M16** | Comunicação emergente | Dois agentes inventam uma língua; acaso→100% (atenção conjunta) |
| **M17** | Composicionalidade | Código composicional **generaliza** p/ combinações nunca vistas (100% vs 0%) |
| **M18** | Grounding | Símbolos referem categorias PERCEBIDAS; comm limitada pelo que se distingue (r=0.90) |

Precedida de estudo da literatura ([09](09-estudo-grounding.md): Harnad, Steels,
cross-situational learning). *Achados honestos:* o contexto temporal ajuda de forma
modesta (M14); compositionalidade exige estrutura/capacidade dada (M17); o grounding
revela que **a percepção limita a linguagem** (M18) — uma verdade real, não um truque.

## O "livro-razão" da honestidade (o que retiramos)

Esta é a assinatura do projeto — afirmações que recuamos quando os dados não
sustentaram, antes que virassem hype:
- M4: "piso do ruído" → fórmula corrigida (ótimo de subespaço).
- M5: "estágios emergem da curiosidade" → vêm da dificuldade; só o currículo é da política.
- M5/M7: "curiosidade vence o acaso" → não, em mundo fácil; vence em *fuga do ruído*.
- M6: "vetorizar escala" → resultado negativo (gargalo é nº de updates).
- M16: regra só-recompensa travava → atenção conjunta resolveu.
- M17: holístico não generaliza (0%) → só o composicional (produtividade).

## O que está genuinamente conquistado vs. aberto

**Conquistado (em miniatura, testado):** todos os primitivos de uma mente —
percepção, plasticidade, embodiment, previsão, curiosidade, memória, conceitos,
spiking, planejamento, visão, comunicação composicional e ancorada — e a sua
integração num organismo.

**Aberto (honesto):**
1. Unir o substrato **spiking** (M10) ao organismo integrado (M13) ao mesmo tempo.
2. **Escala** real — NumPy/CPU é teto; pede GPU/JAX ou neuromórfico.
3. **Linguagem plena** — recursão, semântica profunda, diálogo; M14–M18 são protos.
4. **Mundos ricos** — saímos de grids/barras para ambientes/sensores realistas.

## Até onde dá pra chegar (avaliação franca)

Três níveis, sem ilusão:

- **Já alcançado (projeto solo, agora):** a demonstração completa e honesta do arco
  inteiro, em miniatura — o que ninguém integrou. *Isto é a conquista real.*
- **Alcançável com esforço/recursos modestos:** escalar para GPU, mundos mais ricos,
  unir spiking ao organismo → um agente robusto **nível "animal simples"** que
  genuinamente aprende do zero, percebe, lembra, planeja e comunica com grounding.
  É um programa de pesquisa de verdade, fundável.
- **Fora do alcance solo (a verdade):** cognição e **linguagem humana plenas**,
  emergentes do paradigma puro. **Ninguém chegou lá** (nem os labs financiados);
  exigiria recursos enormes *e* inovações não resolvidas — ou um **híbrido** (núcleo
  cerebral + módulo de linguagem). O "100%" é a **estrela-guia, não a entrega**.

Tradução honesta: o brAIn não vai virar um humano digital sozinho. Mas já é uma
**prova de conceito completa e rara** do paradigma cerebral — e um mapa claro (com
custos reais) de como escalá-lo. Esse é um destino digno e verdadeiro.

## Conclusão

Partimos de uma intuição numa pasta vazia e chegamos a um organismo que percebe,
prevê, lembra, planeja, conceitua, vê e **conversa sobre o que aprendeu a perceber**
— tudo do zero, sem backprop, no paradigma do cérebro. Cada passo medido, cada
exagero recuado. Não é o fim da estrada (o "100%" segue distante e incerto), mas é
um marco real numa fronteira que a ciência deixou aberta — e foi percorrido com
honestidade do primeiro neurônio à primeira palavra ancorada.
