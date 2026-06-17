# 03 — Roadmap

Marcos do projeto. Cada um termina com um **artefato rodável** e uma **pergunta respondida**. Avançamos só quando o anterior está sólido — ciência incremental.

---

## M0 — Fundação ✅ (concluído)
**Objetivo:** definir o terreno antes de codar.
- [x] Visão + hipótese falsificável ([01](01-visao-e-hipotese.md))
- [x] Estado da arte honesto ([02](02-estado-da-arte.md))
- [x] Fundações matemáticas ([04](04-fundacoes-matematicas.md))
- [x] Decisão de stack (NumPy puro → Brian2) e estrutura de repositório
- [x] Ambiente Python + reprodutibilidade (seed fixa, requirements.txt)

**Pergunta respondida:** *o que vamos construir e por quê?*

---

## M1 — Um neurônio vivo ✅ (concluído)
**Objetivo:** simular **um** neurônio Leaky Integrate-and-Fire (LIF) do zero e enxergar seus disparos.
- [x] Implementar a EDO do LIF em NumPy (integração de Euler) — `src/brain/neuron.py`
- [x] Injetar corrente; observar acúmulo de potencial, disparo e reset
- [x] Visualizar: potencial de membrana + curva f-I + raster — `experiments/m1_living_neuron.py`
- [x] Validar contra a teoria (curva f-I bate com a fórmula analítica) — 6/6 testes em `tests/`

**Artefato:** `experiments/output/m1_living_neuron.png` + suíte de testes.
**Pergunta respondida:** *sim — a matemática de um neurônio vive na tela, e bate com a teoria.*

---

## M2 — Uma sinapse que aprende (STDP) ✅ (concluído)
**Objetivo:** dois+ neurônios conectados que **mudam** a força da conexão por timing.
- [x] Implementar STDP online por traços (potenciação/depressão) — `src/brain/synapse.py`
- [x] Demonstrar condicionamento pavloviano *in silico* — `experiments/m2_pavlov.py`
- [x] Medir o peso sináptico mudando ao longo do treino (sino: 0→7 disparos)
- [x] Controle não-pareado (não aprende) — prova de associatividade
- [x] 5 testes de STDP (direção, limites, sinapse fixa) — 11/11 no total

**Artefato:** `experiments/output/m2_pavlov.png` + testes.
**Pergunta respondida:** *sim — uma regra puramente local aprende uma associação, e só quando ela existe de verdade (controle plano).*

---

## M3 — Corpo e mundo (embodiment) ✅ (concluído)
**Objetivo:** fechar o laço percepção → ação → percepção.
- [x] Ambiente 2D com fonte + corpo de tração diferencial — `src/brain/world.py`
- [x] Sensores (luz→corrente) e atuadores (spikes motores→rodas) — laço fechado
- [x] Cérebro reflexivo: fototaxia (excitação cruzada) + varredura (klinocinese)
      + chegada (inibição por proximidade) — `experiments/m3_embodiment.py`
- [x] Baseline aleatório + varredura de parâmetros (`experiments/_m3_sweep.py`)
- [x] Resultado: **98% de sucesso vs 28% (aleatório)**, distância final 5.3 vs 61.3
- [x] 7 testes do mundo/corpo — 18/18 no total

**Artefato:** `experiments/output/m3_embodiment.png` + testes.
**Pergunta respondida:** *sim — o cérebro move um corpo, sente consequências e
busca a fonte de forma competente, batendo o acaso com folga.*

---

## M4 — A máquina de previsão (predictive coding) ✅ (concluído)
**Objetivo:** o brAIn passa a **prever** sua entrada sensorial e aprender com o erro.
- [x] Camada de predictive coding (Rao & Ballard): inferência + erro — `src/brain/predictive.py`
- [x] Aprendizado por regra LOCAL `ΔW ∝ ε·rᵀ` (sem backprop; numa camada, = -∂E/∂W)
- [x] Curva da "surpresa" caindo: **queda de 79%**, confirmada em **held-out**
      (W congelado ≈ erro de treino → é aprendizado, não decoreba); chega perto do
      **ótimo de subespaço** σ²(D−d) ≈ 0.202, ficando logo acima — não "no piso"
- [x] Estrutura emergente: campos receptivos recuperam os padrões do mundo
      (cosseno 0.85–0.97), batem o baseline trivial (k-means colapsa em 5/8)
- [x] 6 testes (inferência reduz erro, surpresa cai, estrutura emerge, resíduo ruidoso
      aproxima σ²(D−d)) — `tests/`

**Artefato:** `experiments/output/m4_predictive.png` + testes.
**Pergunta respondida:** *sim — prever o mundo organiza a percepção sozinho;
campos receptivos emergem sem serem programados, só minimizando a surpresa.*

> ⚠️ **Lacuna de integração (honestidade):** M1–M4 validam cada ingrediente
> **isoladamente**. O M4 é *rate-based* (contínuo) — **não usa spiking nem STDP** —
> e não compartilha código com os módulos M1–M3. O sistema **integrado** que a H1
> descreve (spiking + STDP + predição, corporificado) **ainda não existe**; uni-los
> é trabalho futuro (e as ablações "tirar STDP / predição" do M6 só fazem sentido
> depois disso). "Avançar só com fundação sólida" vale por ingrediente, não como
> uma pilha única já costurada.

---

## M5 — Desenvolvimento (o bebê que cresce) ✅ (concluído, com ressalvas honestas)
**Objetivo:** testar a **H2** — emergência de estágios de desenvolvimento.
- [x] Motivação intrínseca por *learning progress* (Oudeyer) — `src/brain/curiosity.py`
- [x] **Currículo ativo emerge da política** (efeito real): a atenção migra sozinha
      fácil→médio→difícil e abandona o ruído (painel b, todo pós-warmup)
- [x] Imunidade à "TV chiando": novelty-seeking colapsa (30%, **98% preso no ruído**),
      curiosidade por learning-progress resiste (91%, ~18% no ruído) — resultado grande e robusto
- [x] Teste de **esquecimento catastrófico**: real (+0.100 de erro no fácil ao
      aprender o difícil) — `experiments/m5_development.py`
- [x] 5 testes — `tests/test_curiosity.py` (total: **29 verdes**)

**Ressalvas honestas (corrigidas após verificação adversarial — achados reais):**
- ⚠️ **Os "estágios" NÃO emergem da curiosidade.** A ordem fácil→médio→difícil vem da
  **dificuldade intrínseca** (1/3/6 padrões): o aleatório reproduz a mesma ordem
  (15/15 sementes) e até treino isolado sem política nenhuma reproduz. A curiosidade,
  aliás, **quebra** a escada (só ~1/14 sementes mantêm a ordem monotônica), porque
  abandona o difícil quando o progresso satura. O que a política realmente cria é o
  **currículo ativo** (painel b) + a fuga do ruído.
- **Curiosidade NÃO supera o aleatório** nesta tarefa fácil: domínio final 91% vs 97%;
  a difícil é alta variância (≈76±18% curiosidade vs 95±1% aleatório). A alocação
  esperta só compensaria com prática focada cara e/ou muito mais ruído que estrutura.
- **Achado profundo (c×d, a confirmar dentro de uma só corrida):** a curiosidade abandona
  o que domina, e o abandonado *decai*; o aleatório intercala tudo → ensaio incidental →
  não esquece. **Curiosidade sozinha não basta — falta memória/ensaio.** Problema aberto.
- **Escopo da H2:** M5 testa a forma **genérica** ("estágios surgem por dificuldade
  crescente"), não a escada biológica específica (reflexo→sensório-motor→antecipatório→
  objetivo) da [H2](01-visao-e-hipotese.md) — essa depende da integração M1–M4.

**Artefato:** `experiments/output/m5_development.png` + testes.
**Pergunta respondida:** *parcialmente — o currículo ativo e a imunidade ao ruído
EMERGEM da curiosidade; a ordem dos estágios vem da dificuldade; e o desenvolvimento
pleno esbarra em memória/esquecimento (próximo problema).*

---

## M6 — Escala e estudo científico ✅ (concluído)
**Objetivo:** crescer com rigor e extrair conhecimento.
- [x] **Ablação** do predictive coder — `experiments/m6_ablations.py`: aprendizado,
      inferência e capacidade são *load-bearing*; não-negatividade e prior L2 são
      refinamentos. Cada peça justifica sua existência (regra nº3).
- [x] **Escala/vetorização** — `experiments/m6_scale.py` + `learn_batch`: 14–24× de
      throughput, **mas resultado negativo honesto** — minibatch é menos eficiente
      por amostra, então fica mais lento no total. Gargalo = nº de updates, não
      throughput. Escala real pede GPU (modelos grandes) ou neuromórfico.
- [x] **Relatório científico** consolidado — [06-relatorio.md](06-relatorio.md)
- [x] Ablação de política do M5 (curiosidade vs novelty vs aleatório) já no M5
- [x] 30 testes verdes; base reprodutível (seeds fixas, `requirements.txt`)

> Nota: a ablação "tirar STDP / tirar predição" no MESMO agente integrado não foi
> possível — esse agente ainda não existe (lacuna de integração, ver
> [06-relatorio.md](06-relatorio.md)). Ablamos cada módulo isolado.

**Artefato:** `experiments/output/m6_ablations.png` + `06-relatorio.md` + base de código.
**Pergunta respondida:** *que os ingredientes do paradigma cerebral funcionam
isolados, que aprendizado/inferência/capacidade são essenciais, e que integração,
memória e escala continuam abertos — um mapa honesto, não um cérebro pronto.*

---

### Regras do jogo
1. **Mede tudo.** Sem métrica, não houve experimento.
2. **Baseline sempre.** Comparar com agente aleatório/trivial.
3. **Ablação.** Toda peça precisa justificar sua existência.
4. **Reprodutível.** Seed fixa, ambiente versionado, resultados re-geráveis.
5. **Avança só com fundação sólida.** Nada de pular marco.
