# 01 — Visão e Hipótese

## A visão

Construir um agente artificial que adquire comportamento estruturado **a partir do zero**, vivendo num ambiente, usando os mecanismos que sabemos (ou suspeitamos) serem os do cérebro biológico — e **não** o paradigma Transformer/backpropagation.

A pergunta-mãe do projeto:

> *Quanto comportamento inteligente pode emergir de uma rede de neurônios que aprende apenas com regras locais e com o imperativo de prever seu próprio mundo sensorial — sem nenhum conhecimento pré-instalado?*

## Por que isso é diferente do que existe

| Eixo | Transformer (IA atual) | brAIn (paradigma cerebral) |
|---|---|---|
| Quando aprende | Uma vez, no treino | O tempo todo, vivendo |
| Pesos pós-treino | Congelados | Sempre plásticos |
| Sinal de aprendizado | Backprop (gradiente global) | Regras **locais** (STDP/Hebbian) |
| Conhecimento inicial | Toda a internet | **Nada** |
| Relação com o mundo | Texto desencarnado | Corpo + sensores + ação |
| Objetivo | Prever o próximo token | Prever o próximo *estado do mundo* |
| Tempo | Ausente (posicional) | Central (disparos têm timing) |

## A hipótese científica (falsificável)

> **H1.** Uma rede de neurônios *spiking* governada **somente** por (a) plasticidade local dependente de timing (STDP) e (b) um objetivo de minimização de erro de previsão (predictive coding), quando corporificada num laço sensório-motor, adquire representações e comportamento úteis **sem qualquer pré-treino nem backpropagation**.

E a parte ousada, também falsificável:

> **H2.** Essa aquisição segue **estágios de desenvolvimento** análogos aos de um organismo biológico (primeiro controle motor reflexo → coordenação sensório-motora → previsão antecipatória → comportamento dirigido a objetivos), e tais estágios emergem *sem serem programados*, apenas pela dinâmica de aprendizado.

### Como falsear (critérios de fracasso)

A hipótese está **errada** se, depois de implementada com fidelidade:
- A rede não supera um agente de baseline aleatório em nenhuma tarefa do ambiente; **ou**
- O aprendizado exige, na prática, um sinal global tipo-backprop pra funcionar; **ou**
- O comportamento não melhora com o tempo de "vida" (sem curva de aprendizado).

### Como medir sucesso (métricas)

1. **Erro de previsão sensorial** cai ao longo da vida do agente (curva descendente).
2. **Desempenho na tarefa** (ex.: alcançar alvo, evitar dano) sobe acima do baseline aleatório.
3. **Emergência de estrutura**: surgem grupos de neurônios seletivos a estímulos (campos receptivos) que ninguém programou.
4. **Estabilidade**: aprende coisas novas sem esquecer catastroficamente as antigas (teste de *continual learning*).

## A estrela-guia distante: cognição e comunicação EMERGENTES

O norte último do brAIn — o "100%" — não é só perceber e agir. É que **raciocínio
e linguagem NÃO sejam programados, e sim APRENDIDOS**, emergindo do
desenvolvimento, como num cérebro. A escada (cada degrau nasce do anterior):

```
sentir + agir → prever o mundo (M4) → formar conceitos →
  PENSAR (rodar a previsão "por dentro", sem agir: imaginar/simular/planejar) →
    RACIOCINAR (encadear conceitos) → COMUNICAR (grudar símbolos nos conceitos)
```

Insight unificador: no predictive coding, **pensar é o cérebro rodando seu modelo
do mundo internamente** — a mesma máquina do M4 é a semente do pensamento.
**Comunicação não é colada; emerge** da cognição (símbolo aponta para o conceito
vivido — *symbol grounding*). Por isso a linguagem é o caminho mais difícil e
distante, mas coerente com a via puramente cerebral.

**Honestidade:** isto é a inteligência geral pela via biológica. O cérebro humano
prova ser possível, mas entre M4 e "ele raciocina e conversa" há saltos não
resolvidos por ninguém (abstração, memória de longo prazo, composicionalidade,
escala). M1–M5 constroem o *substrato*; raciocínio emergente fica muito além do M5.
Ver também [05-prior-art.md](05-prior-art.md) (ninguém entregou a integração) e o
teto de capacidades em [03-roadmap.md](03-roadmap.md).

### Estratégia de modalidades (decisão de arquitetura, pós-M5)
- **Imagens**: a praia natural deste paradigma — predictive/sparse coding nasceu na
  visão; o M4 já fez emergir detectores de borda. Visão é alcançável no caminho puro.
- **Texto/linguagem**: emergente via grounding (puro, difícil/distante) **ou** via
  módulo pronto/LLM acoplado ao núcleo cerebral (híbrido). Decisão adiada para
  depois do núcleo (M5/M6) estar de pé.

## O que este projeto NÃO promete (no curto prazo / nesta escala)

- Não promete consciência, senciência ou "vida".
- Não promete raciocínio ou linguagem **agora**: são a estrela-guia emergente
  (acima), não entregas dos próximos marcos.
- Não promete igualar GPT/Claude em texto — paradigma diferente; texto puro é a
  praia dos transformers.
- Não promete escala de cérebro humano. Promete o *paradigma* certo, em escala
  honesta e crescente.

## Princípio norteador

> Preferir sempre o mecanismo **biologicamente plausível** ao mecanismo conveniente — mesmo que mais lento — e medir tudo. Ciência, não mágica.
