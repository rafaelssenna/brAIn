# 02 — Estado da Arte (honesto)

Mapa do que a ciência já alcançou nos campos que o brAIn toca. Serve para (a) não reinventarmos a roda e (b) sermos realistas sobre os limites.

## 1. Por que os Transformers não são "o cérebro"

O Transformer (Vaswani et al., 2017, *Attention is All You Need*) é extraordinário, mas:
- Aprende via **backpropagation**, considerada biologicamente implausível — não há mecanismo neural conhecido que transporte o gradiente exato de volta pelas camadas (o "weight transport problem").
- Os pesos **congelam** após o treino. Não há aprendizado contínuo durante o uso.
- Não tem **tempo** intrínseco nem **corpo**. Opera sobre tokens, não sobre um fluxo sensório-motor.

Conclusão: ótimo para linguagem, errado como modelo de *como um cérebro aprende a viver*.

## 2. Conectômica — o "mapa" do cérebro

- **C. elegans** (verme): 302 neurônios, conectoma conhecido desde os anos 1980. Projeto **OpenWorm** tenta simulá-lo inteiro — ainda incompleto.
- **Drosophila** (mosca-da-fruta): conectoma cerebral completo (~140 mil neurônios, ~50 milhões de sinapses) publicado pelo **FlyWire** em 2024 — o maior cérebro totalmente mapeado até hoje.
- **Camundongo**: apenas fragmentos do córtex.
- **Humano**: ~86 bilhões de neurônios, ~10¹⁴ sinapses. Mapeado: quase nada em escala sináptica.

**Implicação para o brAIn:** não há blueprint humano para copiar. Devemos *gerar* a arquitetura por regras de desenvolvimento, não copiá-la.

## 3. Neurônios spiking (SNN) e hardware neuromórfico

- **Modelos de neurônio** (ver doc 04): Hodgkin-Huxley (1952, biologicamente fiel, caro), Izhikevich (2003, rico e barato), Leaky Integrate-and-Fire / LIF (simples, padrão de partida).
- **Hardware neuromórfico**: Intel **Loihi 2**, IBM **TrueNorth**, **SpiNNaker** (Manchester). Executam SNNs com baixíssima energia. Não precisamos deles para começar — simulamos em CPU/GPU.

## 4. Regras de aprendizado biologicamente plausíveis

- **Hebbian** ("neurons that fire together wire together"): a regra mais antiga (Hebb, 1949).
- **STDP** (Spike-Timing-Dependent Plasticity): a sinapse fortalece se o pré-sináptico dispara *antes* do pós-sináptico, e enfraquece no caso contrário. Base experimental sólida (Bi & Poo, 1998).
- **Regras de três fatores**: STDP modulada por um sinal global (ex.: dopamina) — ponte plausível para aprendizado por recompensa.
- **Predictive coding** (Rao & Ballard, 1999): o córtex aprende minimizando *erro de previsão*; pode aproximar backprop **com sinais locais** — área quente de pesquisa (candidato para resolver a implausibilidade do backprop).
- **Free Energy Principle** (Karl Friston): grande teoria unificadora — o cérebro minimiza "energia livre variacional", i.e., surpresa. Matematicamente pesada, ver doc 04.

## 5. O problema do esquecimento catastrófico

Redes neurais comuns, ao aprender algo novo, **apagam** o que sabiam (catastrophic forgetting). O cérebro não faz isso. *Continual / lifelong learning* é o campo que ataca o problema (EWC, replay, modularidade). É um critério de sucesso central do brAIn.

## 6. Robótica do desenvolvimento (Developmental Robotics)

Campo que estuda agentes que aprendem como bebês: **motivação intrínseca** (curiosidade, novidade), aprendizado por estágios, exploração ativa. Nomes: Pierre-Yves Oudeyer, Juergen Schmidhuber (curiosity/compression). É o molde teórico do nosso M5.

## 7. Ferramentas práticas (Python)

| Ferramenta | Uso |
|---|---|
| **Brian2** | Simulador de SNN expressivo, ótimo para pesquisa/prototipagem |
| **NEST** | Simulador de SNN de larga escala |
| **snnTorch / Norse** | SNN sobre PyTorch (GPU, treináveis) |
| **Nengo** | Framework do *Neural Engineering Framework* (Eliasmith) |
| **BindsNET** | SNN + aprendizado por reforço |
| **NumPy / JAX** | Para implementar do zero e entender cada equação |

**Decisão de partida (a confirmar no M1):** implementar o primeiro neurônio **do zero em NumPy** (para entender a matemática na pele) e migrar para **Brian2** quando a rede crescer.

## Resumo do que é e não é viável hoje

| Pergunta | Resposta honesta |
|---|---|
| Copiar o cérebro humano? | **Não.** Sem mapa, sem algoritmo, sem compute. |
| Aprender do zero, online, sem backprop, embodied? | **Sim**, em pequena escala. É o coração do brAIn. |
| Ver comportamento e estrutura emergirem? | **Sim**, é realista e mensurável. |
| Rodar num PC comum no começo? | **Sim.** Escala vem depois. |
