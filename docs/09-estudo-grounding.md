# 09 — Estudo: Symbol Grounding (preparação do M18)

Antes de construir o M18 (comunicação ancorada), estudei a literatura. Este
documento resume o que importa e fixa um design fundamentado. Fontes ao final.

## O problema (por que o M16/M17 ainda não "significa" nada)

**Symbol Grounding Problem** (Harnad, 1990): como o significado de um símbolo pode
se ancorar em algo que não seja apenas outros símbolos sem sentido? Hoje, no
M16/M17, os agentes comunicam sobre **ids/atributos abstratos que NÓS entregamos**
— os símbolos são fichas, não significam nada *para o agente*. É o problema de
Harnad em miniatura.

**Solução de Harnad:** símbolos devem ser ancorados *bottom-up* em representações
não-simbólicas: (a) **icônicas** (a projeção sensorial) e (b) **categóricas**
(detectores de feature disparados por invariâncias dos objetos). *O significado de
um símbolo elementar é a CATEGORIZAÇÃO da experiência sensorial.* 

→ **Insight central para nós:** as "representações categóricas" de Harnad são
exatamente o que o **M9 (predictive coder hierárquico)** já produz — categorias
invariantes formadas da percepção. Então o brAIn JÁ TEM onde ancorar os símbolos:
nos conceitos que ele forma sozinho. Falta ligar a comunicação (M16) a eles.

## O modelo canônico: language games (Steels)

O **Talking Heads Experiment** (Steels, 1999–2001) é o trabalho de referência:
agentes corporificados percebem cenas reais (figuras via câmeras) e, jogando
"language games", fazem **um léxico compartilhado EMERGIR sem controle central** —
com os símbolos ancorados na percepção. O loop:

1. **Contexto**: ambos os agentes percebem um conjunto de objetos.
2. **Jogo de discriminação** (conceitualização): o falante escolhe um objeto-tópico
   e acha uma CATEGORIA perceptual que o distingue dos demais (cria/refina
   categorias quando precisa).
3. **Jogo de nomeação/adivinhação**: o falante produz uma palavra para a categoria
   (inventa se não há); o ouvinte interpreta palavra→categoria→aponta o objeto.
4. **Feedback (atenção conjunta)**: o falante revela o tópico real. Acertou →
   ambos reforçam a associação palavra↔categoria; errou → o ouvinte adota/ajusta.
5. Repetição → léxico se auto-organiza, **ancorado em categorias perceptuais**.
   Converge por reforço da associação usada + inibição lateral das concorrentes.

→ Isto valida e estende o que já fizemos: o feedback de atenção conjunta é o mesmo
mecanismo *cross-situacional* que destravou o M16. O que falta é o passo 1–2: os
conceitos vêm da PERCEPÇÃO do agente, não de ids dados.

## O que a versão moderna confirma

- **Emergent communication com entrada perceptual** (Lazaridou et al., 2018):
  agentes aprendem protocolos a partir de pixels; achado-chave — *linguagem
  composicional estruturada emerge mais provavelmente quando os agentes percebem o
  mundo como estruturado*. → Casa com o M17: estrutura na percepção → estrutura na
  língua. Logo, ancorar nos conceitos compostos do M9 favorece composicionalidade.
- **Cross-situational learning** (aquisição de palavras em bebês/robôs): aprende-se
  o mapeamento palavra↔percepto pela co-ocorrência ao longo de situações. → É
  literalmente a regra de "atenção conjunta" do M16. Já estamos no caminho certo.

## Design recomendado para o M18

**Comunicação ancorada via language game:** dois agentes que **percebem e
categorizam** os objetos do mundo (via representação tipo-M9) e desenvolvem um
léxico compartilhado para falar **sobre suas próprias categorias percebidas**.

- **Mundo**: K objetos (padrões). Cada agente identifica o objeto pela sua
  representação perceptual aprendida (percepção → categoria), não por um id dado.
- **Jogo de referência**: contexto = alvo + distratores; falante percebe→categoriza
  o alvo→emite símbolo (signaling do M16); ouvinte decodifica símbolo→sua categoria
  →escolhe o objeto; feedback de atenção conjunta; reforço cross-situacional.
- **Ancoragem**: o símbolo passa a referir uma CATEGORIA PERCEPTUAL (Harnad), não
  uma etiqueta. Demonstra-se mostrando que (i) o agente identifica o objeto pela
  percepção e (ii) a comunicação só funciona porque os símbolos grudam nessas
  categorias compartilhadas.
- **Métrica**: acurácia de comunicação sobe; e — teste honesto — a comunicação
  **degrada se as representações perceptuais dos dois agentes não se alinham**
  (grounding real depende de conceitos compatíveis). Esse é o achado interessante.

**Honestidade:** continua escala de brinquedo; não é linguagem plena. Mas é a
diferença entre "jogo de fichas" e "símbolos que significam algo que o agente
aprendeu vivendo" — o primeiro grounding de verdade.

## Fontes
- Harnad (1990), *The Symbol Grounding Problem* —
  https://www.cs.ox.ac.uk/activities/ieg/e-library/sources/harnad90_sgproblem.pdf ·
  Scholarpedia: http://www.scholarpedia.org/article/Symbol_grounding_problem
- Steels, *Language games for autonomous robots* / *The Talking Heads Experiment* —
  https://ai.vub.ac.be/sites/default/files/steels-03c.pdf
- Lazaridou et al. (2018), *Emergence of Linguistic Communication from Referential
  Games with Symbolic and Pixel Input* — https://arxiv.org/pdf/1804.03984
- Cross-Situational Learning with Bayesian Generative Models (Taniguchi et al.) —
  https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2017.00066/full
