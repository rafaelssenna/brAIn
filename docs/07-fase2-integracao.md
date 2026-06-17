# 07 — Fase 2: O Organismo Integrado (M7→M12 e além)

A Fase 1 (M1–M6) provou que **os ingredientes funcionam isolados**. A Fase 2 tem
um objetivo só, e é o maior buraco do mundo (ninguém preencheu — ver
[05-prior-art.md](05-prior-art.md)): **costurar os ingredientes num único
organismo** que nasce ignorante, percebe, age, prevê, é curioso, e — degrau a
degrau — caminha rumo à cognição emergente da [visão](01-visao-e-hipotese.md).

> Princípio que não muda: mecanismo biologicamente plausível, mede tudo, baseline
> sempre, ablação, reprodutível, **verificação adversarial a cada marco**, e
> recuar onde os dados não sustentam. Ciência, não mágica.

## A decisão central: o substrato

A Fase 1 deixou uma fratura honesta: M4 (predição) é *rate-based*; M2/M3
(STDP/spiking) são spiking. Integrar exige UM substrato. Há dois caminhos:

- **(A) Purista:** tudo em neurônios spiking com plasticidade local (spiking
  predictive coding). É o ideal biológico — e uma fronteira de pesquisa dura.
- **(B) Faseado (recomendado):** integrar primeiro os módulos *rate-based* já
  provados (M3+M4+M5) num laço que funciona, e **depois** portar para spiking,
  quando a arquitetura estiver entendida.

**Recomendação:** começar por (B) e migrar para (A) no M10. Razão: integração já
é arriscada; somar a dificuldade do spiking predictive coding de uma vez é receita
de fracasso não-diagnosticável. Provar o laço primeiro, depois trocar o substrato.

## Os marcos

### M7 — O laço integrado (active inference mínima) ✅ (concluído)
**Objetivo:** fundir M3 (corpo) + M4 (predição) + M5 (curiosidade) em UM agente.
- [x] `src/brain/agent.py`: `RingWorld` (anel de lugares) + `IntegratedAgent`
      (corpo + predictive coder + curiosidade) — um laço só.
- [x] O modelo de mundo aprende online (predictive coding local) os padrões dos
      lugares; a curiosidade (learning progress) guia a **navegação física**.
- [x] 4 testes — `tests/test_agent.py` (total: **34 verdes**)

**Resultados (`experiments/m7_integrated.py`):**
- ✅ **O laço fecha e funciona**: aprende o modelo de mundo até **94%** de domínio.
- ✅ **Evita fisicamente o ruído**: 6% do tempo no ruído vs **25%** do aleatório (4×).
- ✅ **Aprende mais rápido no começo** que o passeio aleatório (curva mais íngreme).

**Ressalvas honestas:**
- No domínio **final**, o passeio aleatório alcança 99% vs 94% — num anel pequeno,
  vagar cobre tudo, e o curioso perde um pouco por abandonar lugares (esquecimento
  → alvo do **M8**). A vitória da curiosidade é velocidade + fuga do ruído, não
  domínio assintótico (consistente com o achado do M5).
- Implementação é **active inference em espírito**, não o modelo ação-condicionado
  completo (prevê a sensação por lugar e navega por learning progress; não prevê
  ainda a consequência exata de cada ação). Substrato **rate-based** (spiking = M10).

**Pergunta respondida:** *sim — os módulos viram UM organismo que percebe, prevê,
é curioso e age; e o corpo torna a curiosidade consequente (fuga do ruído 4×).*

### M8 — Memória: vencer o esquecimento ✅ (concluído)
**Objetivo:** fechar a lacuna do M5/M7 (a curiosidade abandona e o abandonado decai).
- [x] `src/brain/memory.py`: `ReplayBuffer` com **amostragem por reservatório**
      (mantém amostra uniforme de TODA a vida) + replay opcional no `IntegratedAgent`.
- [x] 4 testes — `tests/test_memory.py` (total: **38 verdes**)

**Resultados (`experiments/m8_memory.py`) — H4 CONFIRMADA:**
- ✅ **Esquecimento quase zerado**: erro no conjunto antigo na fase nova vai de
      **+0.436 (sem replay) → +0.004 (com replay)**.
- ✅ **Replay DESTRAVA a curiosidade**: no organismo integrado, curioso+replay sobe
      de 93% → **99% de domínio, empatando com o aleatório** — *e* ainda evita o
      ruído 3× melhor (9% vs 25%). Pela 1ª vez o agente curioso+memória domina o
      quadro geral (mesmo aprendizado, muito menos desperdício).

**Mecanismo:** reensaio de experiências guardadas (replay hipocampal do sono). A
amostragem por reservatório é o que impede esquecer o antigo (um FIFO falharia).
**Pergunta respondida:** *sim — replay vence o esquecimento catastrófico, e isso
era exatamente o que faltava para a curiosidade compensar.*

### M9 — Hierarquia: rumo a conceitos ✅ (concluído)
**Objetivo:** empilhar camadas de predictive coding (Rao & Ballard: cada camada
prevê a de baixo, só o erro sobe; crédito LOCAL por camada, sem "weight transport").
- [x] `src/brain/hierarchy.py`: `HierarchicalPredictiveCoder` (64→16→6), inferência
      e aprendizado **locais por camada** — nenhum backprop entre camadas.
- [x] 4 testes — `tests/test_hierarchy.py` (total: **42 verdes**)

**Resultados (`experiments/m9_hierarchy.py`) — H5 parcialmente sustentada:**
- ✅ **Invariância (abstração) EMERGE**: exemplares de uma mesma categoria colapsam
      para quase o MESMO código no topo (similaridade dentro-da-categoria
      **0.81 → 0.92 → 0.99** da entrada ao topo). A matriz de similaridade do topo
      é nitidamente bloco-diagonal por categoria — o topo trata exemplares variados
      como "a mesma coisa". Isso é o que um *conceito* é.
- ✅ Demonstra o ganho do PC sobre o backprop: aprendizado puramente local por camada.

**Ressalva honesta:** a **separação líquida** (dentro − entre) NÃO cresce com a
profundidade (≈0.77→0.76) — a entrada já era separável (blocos ~ortogonais) e os
códigos não-negativos do topo se sobrepõem um pouco (entre = 0.23). O ganho real é
a **invariância**, não separabilidade extra. (Risco previsto do PC profundo: exigiu
ajuste de `n_infer`/`eta_r`; estável aqui, mas frágil em escala maior.)
**Pergunta respondida:** *sim — conceitos (categorias invariantes) emergem nas
camadas altas, sem serem programados; é a semente da abstração.*

### M10 — Substrato spiking unificado (o salto purista) ✅ (concluído — com escopo honesto)
**Objetivo:** portar o aprendizado para neurônios spiking com plasticidade local.
- [x] `src/brain/spiking_predictive.py`: `SpikingPredictiveCoder` — unidades
      latentes são **neurônios LIF (M1)** reais; predição aprendida por regra
      **LOCAL** `ΔW ∝ ε·ρᵀ` sobre a TAXA DE DISPARO. Sem backprop.
- [x] 4 testes — `tests/test_spiking_predictive.py` (total: **46 verdes**)

**Resultado (`experiments/m10_spiking.py`):**
- ✅ **A rede spiking APRENDE**: surpresa cai **0.77 → 0.14 (−82%)**; códigos
      spiking específicos por padrão (raster); campos receptivos emergem.
- ✅ **A fratura M4 (rate vs spiking) está costurada**: predictive coding +
      neurônios spiking + plasticidade local numa rede só.
- 🔧 Risco materializou-se e foi resolvido honestamente: a inferência **oscilava**
      (corrigida com integrador amortecido) e o resíduo era alto (corrigido com
      ganho de entrada maior — a teoria do "código vazado" previu isso).

**Escopo honesto (o que NÃO é, ainda):** é **rate-coded** (o erro é lido em taxa,
não em spikes temporais como o STDP do M2); é **pequeno e lento**; e **ainda não
está integrado ao corpo/curiosidade** (M7). Ou seja: o ingrediente mais difícil da
**H1** (predição aprendida num substrato spiking local) agora existe — mas a H1
plena (spiking + STDP temporal + predição + corporificado, tudo junto) ainda pede
a costura final. Grande passo, não a chegada.
**Pergunta respondida:** *sim — predictive coding roda em neurônios spiking com
regra local; o substrato purista é viável.*

### M11 — Agência: o primeiro "pensar" ✅ (concluído)
**Objetivo:** o agente usa o modelo de mundo para **simular futuros** e escolher
ações — planejar = prever por dentro.
- [x] `src/brain/planning.py`: `WorldModel` (transição aprendida vivendo), `plan`
      (busca na imaginação), `GridWorld` com parede + agente reativo (baseline).
- [x] 4 testes — `tests/test_planning.py` (total: **50 verdes**)

**Resultados (`experiments/m11_planning.py`) — H6 CONFIRMADA:**
- ✅ **Planejar vence reagir**: sucesso **100% vs 83%**; e o planejador chega em
      **5.2 passos vs 15.2** (caminhos ótimos, 3× mais eficiente).
- ✅ **Imaginação fiel**: o caminho imaginado no modelo == o caminho real executado
      (idênticos) — o agente pode confiar no que imagina.
- ✅ **Desvio por antecipação**: contorna uma parede que o reativo nunca contorna
      (ele trava e oscila) — porque imagina o caminho antes de andar.

**Significado:** "pensar = rodar a previsão por dentro" deixou de ser metáfora — a
MESMA máquina de previsão dos marcos anteriores, agora usada para AÇÃO (imaginar
consequências e deliberar). É a ponte concreta para a cognição emergente (Fase 3).
**Pergunta respondida:** *sim — o agente imagina futuros, planeja, e isso supera
reagir; o primeiro lampejo de pensamento.*

### M12 — Escala + visão ✅ (concluído — Fase 2 COMPLETA)
**Objetivo:** sensores mais ricos (visão, a modalidade natural) e entrada maior.
- [x] `experiments/m12_vision.py`: predictive coder em pedaços de "imagem"
      (bordas orientadas sobrepostas), entrada **144 dim** (12x12, vs 64 antes), 48 latentes.
- [x] 2 testes — `tests/test_vision.py` (total: **52 verdes**)

**Resultados — visão é mesmo a praia do paradigma:**
- ✅ **Detectores de borda emergem**: o dicionário aprendido (colunas de W) vira
      features **localizadas e orientadas** — como células simples de V1. Localização
      média **0.72** (espalhado=0.25). Surpresa **0.77 → 0.08** (−89%). Sem rótulos.
- ✅ Escalou para 144 dim em ~5s (matemática escala bem).

**Honesto sobre escala:** NÃO migrei para JAX/GPU — o M6 já provou que NumPy/CPU
não é o caminho para escala de verdade (gargalo = nº de updates / substrato). A
matemática escala; o *substrato* é que pede GPU ou neuromórfico. Gabors de livro
pedem ainda mais escala + branqueamento — aqui são bordas orientadas localizadas.
**Pergunta respondida:** *sim — alimentado com visão, o brAIn aprende detectores
de borda como o córtex visual; a modalidade natural do paradigma se confirma.*

---

### M13 — A costura final: o organismo num laço só ✅ (concluído)
**Objetivo:** juntar as peças num ÚNICO agente que roda tudo junto.
- [x] `src/brain/integrated.py`: `IntegratedBrain` = corpo num grid com paredes +
      modelo de transição/**planejamento** (M11) + predictive coder **hierárquico**
      (M9) + **curiosidade** que escolhe metas (M5) + **replay** (M8), num laço só.
- [x] 2 testes — `tests/test_integrated.py` (total: **54 verdes**)

**Resultado (`experiments/m13_integrated.py`) — média de 3 seeds:**
- ✅ **Efeito de integração robusto**: o conteúdo C fica atrás de uma parede; só o
      organismo que **planeja** o alcança — **visitas a C: integrado 10.7% vs
      reativo 0.0%** (nunca, em nenhum seed). Domínio total **57% vs 42%**.
- ✅ A divisão de trabalho emerge: curiosidade decide *para onde*, planejamento traça
      *como chegar* (contorna a parede), hierarquia *aprende*, memória *retém*.

**Honesto:** o domínio por-lugar é ruidoso (hierarquia profunda + dinâmica
estocástica), por isso o resultado robusto é medido por visitas a C + média de
seeds. Substrato rate-based (o spiking do M10 fica como substrato futuro).

## ✅ FASE 2 COMPLETA — o organismo integrado existe
M7–M13 entregues, **54 testes verdes**. O brAIn agora é UM organismo que percebe,
prevê, é curioso, age, **lembra** (replay), forma **conceitos** (hierarquia), roda
em **spiking**, **planeja** imaginando futuros, aprende **visão**, e — no M13 —
**costura tudo num laço só**. A integração que ninguém completou
([05](05-prior-art.md)) foi percorrida — em miniatura, honesta, verificada. Falta
unir o substrato spiking ao agente embodied simultaneamente, e a Fase 3
(cognição/linguagem emergentes) — o "100%" distante.

## O horizonte distante (Fase 3, M13+): cognição e linguagem emergentes
Abstração → conceitos → *symbol grounding* → **comunicação emergente**. É a
estrela-guia (o "100%"): raciocínio e linguagem APRENDIDOS, não programados.
- Honestidade brutal: isto é a inteligência geral pela via biológica. Distante,
  incerto, e — para linguagem — provavelmente o ponto onde um **híbrido** (núcleo
  cerebral + módulo de linguagem) faz mais sentido que o purismo. Não é deste ciclo;
  é o norte que justifica cada marco acima.

## Riscos e honestidade
- **Integração pode instabilizar** (M7) e **o spiking pode não escalar** (M10) —
  são pesquisa, não engenharia garantida. Trataremos cada um como hipótese
  falsificável, com plano B explícito (híbrido) quando o purismo travar.
- **Vamos medir o custo** (o Python já é gargalo): M12 pode precisar vir antes se
  o M9 ficar lento demais.
- Cada marco entrega artefato rodável + testes + verificação adversarial. Sem pular.

## Próximo passo imediato
**M7 — o laço integrado.** É o primeiro momento em que o brAIn deixa de ser
"peças numa caixa" e vira *um agente que existe e aprende como um todo*. Tudo que
vem depois depende dele.
