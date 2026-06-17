# 05 — Prior Art: alguém já fez isso?

Pesquisa do estado da arte (mid/late 2025) sobre quem já tentou construir uma IA
que **nasce ignorante e aprende do zero como um cérebro**. Conduzida por busca
multi-agente com fontes. Resposta curta: **ninguém entregou a integração
completa** — cada ingrediente existe isolado e maduro, mas o full-stack não.

## Os "5+ pilares" da nossa visão
1. **Nasce ignorante** (sem pré-treino).
2. **Neurônios spiking** com plasticidade **local** (STDP).
3. **Sem backpropagation.**
4. **Corporificado** num laço sensório-motor.
5. Guiado por **erro de previsão / energia livre**.
6. **Desenvolve-se em estágios**, espelhando a mente humana.

## Ingredientes = prior art (não inventamos nenhum)
| Ingrediente | Referência |
|---|---|
| Neurônios spiking em escala | Spaun / Spaun 2.0 — Eliasmith 2012 (~2,5M neurônios) |
| Plasticidade local STDP | Diehl & Cook 2015; Kheradpisheh et al. 2018 |
| Crédito sem backprop | Predictive coding (Rao & Ballard 1999; Millidge/Bogacz/Salvatori; µPC 2025); Forward-Forward (Hinton 2022); Equilibrium Propagation (Scellier & Bengio 2017) |
| Energia livre / surpresa | Free Energy Principle, Active Inference (Friston; Parr/Pezzulo/Friston 2022) |
| Corporificação | iCub (Metta, Cangelosi); OpenWorm; BAAIWorm; Fujii & Murata |
| Curiosidade / motivação intrínseca | Oudeyer (IMGEP); Schmidhuber (compression progress) |
| Estágios de desenvolvimento | Cognitive/Developmental Robotics (Cangelosi); CDN (Alicea/Parent 2021) |
| Substrato neuromórfico | Intel Loihi 2 / Hala Point; SpiNNaker2; BrainScaleS-2 |
| Continual learning "nasce do zero" | Alberta Plan / OaK (Sutton); "Era of Experience" (Silver & Sutton 2025); loss-of-plasticity (Nature 2024) |
| Conectomas | FlyWire (fly cerebral completo, 2024); Blue Brain; HBP/EBRAINS |

## A prova de que a integração não existe
**Hamburg et al. 2024**, *Entropy* 26(7):582 (Sheffield Hallam) — manifesto de
"developmental neurorobotics" que define o alvo exato (corporificado +
neuromórfico spiking + active inference + estágios de desenvolvimento espelhando
a mente humana) e, após survey, afirma que a integração **não foi feita** — nem
acharam estudo combinando active inference com robôs spiking, quanto mais com
arco de desenvolvimento.

## Os mais próximos (cada um perde ≥2 pilares)
- **BrainCog / BORN** (Yi Zeng, Institute of Automation, CAS): SNN open-source +
  plasticidade local + módulos cérebro-mapeados + robôs, sob visão "AGI viva".
  *Falta:* é kit de módulos separados, não um organismo que nasce ignorante e se
  desenvolve; energia livre não é o objetivo unificador.
- **Numenta / Thousand Brains — Monty** (Jeff Hawkins, Viviane Clay; open-source
  nov/2024): o mais limpo born-ignorant + corporificado + sem backprop, espelha o
  neocórtex (frames de referência, Hebbian). *Falta:* **não é spiking**; energia
  livre só retroajustada; sem estágios de desenvolvimento; coluna é hand-specified.
- **Keen Technologies** (John Carmack + Richard Sutton) / Alberta Plan / OaK: o
  mais próximo em *espírito* — nascer ignorante, corporificado (robô jogando Atari
  real), online, anti-LLM. *Falta:* ainda usa **backprop** (RL); só "sinais de
  vida"; OaK é visão, não agente rodando; não spiking; sem energia livre.
- **Active Inference corporificado** (manifesto Hamburg 2024; Fujii & Murata 2024):
  o objetivo *é* a visão. *Falta:* implementações treinam com **backprop** e
  pré-treino offline; versões neuromórficas puras são toy.
- **BAAIWorm / OpenWorm** (BAAI+PKU, Nature Comp. Sci. dez/2024; OpenWorm): laços
  cérebro-corpo-mundo biológicos reais, quimiotaxia realista. *Falta:* é o
  **inverso** de nascer ignorante — pesos *otimizados* p/ imitar um verme real,
  sem mecanismo de desenvolvimento.

> Ponto crucial: "sem backprop" **E** "desenvolve comportamento amplo" não são
> satisfeitos *juntos* por praticamente ninguém.

## Onde o brAIn se encaixa (honesto)
Território genuinamente aberto, mas a novidade é a **integração e a ambição**, não
nenhuma peça isolada (não reivindicar novidade em mecanismo nenhum). Como projeto
pequeno, estamos **muito abaixo da fronteira financiada** (BrainCog/CAS, Numenta,
Keen, labs neuromórficos) em maturidade e escala. Contribuição crível: **tentar a
integração que cada um deixa incompleta**, apoiando em andaimes existentes
(aprendizado sensório-motor estilo Monty; plasticidade local em substrato
neuromórfico) em vez de reinventar as partes. A visão é sólida e desocupada; o gap
é execução, escala e os problemas duros (eficiência amostral, continual learning
sem esquecer, escalar aprendizado local não-backprop além de brinquedos).

## Bottom line
**Não** — até o fim de 2025 ninguém construiu a visão integrada. Cada ingrediente
existe isolado; a área nomeou o alvo como não-construído (Hamburg 2024); os mais
próximos perdem ≥2 pilares. O cérebro **born-ignorant, spiking, no-backprop,
corporificado, guiado por energia livre e em estágios de desenvolvimento** segue
sendo terra de ninguém.
