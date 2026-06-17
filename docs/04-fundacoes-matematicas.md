# 04 — Fundações Matemáticas

A matemática do brAIn, do neurônio único à teoria geral. Notação completa (você pediu sem simplificar). Cada bloco tem **intuição + fórmula**.

---

## 1. O neurônio como sistema dinâmico

Um neurônio acumula carga (potencial de membrana `V`) e "dispara" (spike) quando cruza um limiar. É uma equação diferencial.

### 1.1 Leaky Integrate-and-Fire (LIF) — nosso ponto de partida

A membrana é um capacitor com vazamento. Intuição: o potencial sobe com a corrente de entrada e "vaza" de volta ao repouso.

$$
\tau_m \frac{dV(t)}{dt} = -\big(V(t) - V_{\text{rest}}\big) + R\, I(t)
$$

- `V(t)` — potencial de membrana
- `V_rest` — potencial de repouso
- `τ_m = R·C` — constante de tempo da membrana (resistência × capacitância)
- `I(t)` — corrente de entrada
- **Regra de disparo:** se `V(t) ≥ V_threshold` → emite spike e `V ← V_reset` (e fica em repouso por um período refratário).

**Integração numérica (Euler explícito), passo `Δt`:**

$$
V_{k+1} = V_k + \frac{\Delta t}{\tau_m}\Big[-(V_k - V_{\text{rest}}) + R\,I_k\Big]
$$

Validação canônica: a **curva f–I** (frequência de disparo em função de `I`) deve crescer monotonicamente acima de um limiar de rheobase.

### 1.2 Hodgkin–Huxley (referência biológica)

O modelo fiel (Nobel 1963). Correntes iônicas de Na⁺ e K⁺ com portões dependentes de voltagem:

$$
C_m \frac{dV}{dt} = I - g_{Na}m^3h\,(V - E_{Na}) - g_K n^4 (V - E_K) - g_L (V - E_L)
$$

com `m, h, n` seguindo cada um `dx/dt = α_x(V)(1-x) - β_x(V)x`. Rico e caro — usamos como referência, não como motor de larga escala.

### 1.3 Izhikevich (o meio-termo)

Reproduz quase toda a diversidade de disparos do córtex com 2 EDOs baratas:

$$
\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I, \qquad \frac{du}{dt} = a(bv - u)
$$

com reset: se `v ≥ 30`mV → `v ← c`, `u ← u + d`. Ótimo candidato quando a rede crescer.

---

## 2. Aprendizado local: plasticidade sináptica

O "peso" `w` de uma sinapse é o quanto o disparo pré-sináptico influencia o pós. Aprender = mudar `w` **sem sinal global**.

### 2.1 Regra de Hebb

$$
\Delta w_{ij} \propto x_i \, x_j
$$

"Disparam juntos, conectam-se." Instável sozinha (cresce sem limite) — precisa de normalização/decaimento.

### 2.2 STDP (Spike-Timing-Dependent Plasticity)

O sinal é o **timing relativo** `Δt = t_post − t_pre`:

$$
\Delta w =
\begin{cases}
+A_{+}\, e^{-\Delta t / \tau_{+}} & \text{se } \Delta t > 0 \quad (\text{pré antes de pós} \Rightarrow \textbf{potencia})\\[4pt]
-A_{-}\, e^{+\Delta t / \tau_{-}} & \text{se } \Delta t < 0 \quad (\text{pós antes de pré} \Rightarrow \textbf{deprime})
\end{cases}
$$

Causalidade vira aprendizado: o que ajudou a causar o disparo é reforçado. **É o coração do M2.**

### 2.3 Regras de três fatores (ponte para recompensa)

STDP modulada por um neuromodulador global `M(t)` (análogo à dopamina):

$$
\Delta w_{ij} = \eta \, M(t) \, e_{ij}(t)
$$

onde `e_ij` é um *traço de elegibilidade* (memória curta da coincidência pré-pós). Permite aprendizado por reforço biologicamente plausível.

---

## 3. Predictive coding — o cérebro que prevê (M4)

**Ideia (Rao & Ballard, 1999):** cada camada tenta **prever** a atividade da camada abaixo; o que sobe pela hierarquia é só o **erro de previsão**.

Erro numa camada `l`:
$$
\varepsilon_l = x_l - \hat{x}_l, \qquad \hat{x}_l = f(W_l\, r_{l+1})
$$

- `x_l` — atividade observada na camada `l`
- `r_{l+1}` — estado/representação da camada de cima
- `\hat{x}_l` — previsão descendente
- `ε_l` — erro (o que a camada de cima não previu)

A rede minimiza a energia (soma dos erros quadráticos, com precisão `Π`):
$$
E = \sum_l \varepsilon_l^\top \, \Pi_l \, \varepsilon_l
$$

As atualizações de `r` e `W` que minimizam `E` usam **apenas quantidades locais** (`ε_l` e `r_{l+1}`) — por isso é a candidata plausível a "backprop do cérebro".

---

## 4. Free Energy Principle (a teoria-mãe)

**Friston:** organismos resistem à desordem minimizando a **surpresa** `−log p(o)` de suas observações `o`. Como a surpresa é intratável, minimiza-se um limite superior, a **energia livre variacional**:

$$
F = \underbrace{D_{\mathrm{KL}}\!\big(q(s)\,\|\,p(s)\big)}_{\text{complexidade}} \; - \; \underbrace{\mathbb{E}_{q(s)}\!\big[\log p(o\mid s)\big]}_{\text{acurácia}}
$$

- `s` — estados ocultos do mundo; `o` — observações
- `q(s)` — crença interna (modelo do agente); `p` — geração do mundo
- Minimizar `F` = ser acurado **sem** ser desnecessariamente complexo.

Predictive coding (seção 3) é uma forma de minimizar `F` por percepção. Minimizar `F` por **ação** (mudar o mundo para bater com a previsão) é a *active inference* — base teórica do embodiment (M3) e do desenvolvimento (M5).

---

## 5. Teoria da informação (régua transversal)

Para medir aprendizado e estrutura:
- **Entropia:** `H(X) = −Σ p(x) log p(x)` — incerteza.
- **Informação mútua:** `I(X;Y) = H(X) − H(X|Y)` — quanto um neurônio "sabe" sobre o estímulo (mede campos receptivos emergentes, M5).
- **Divergência KL:** `D_KL(q‖p)` — distância entre crença e realidade (aparece em `F`).

---

## 6. Mapa: qual matemática em qual marco

| Marco | Matemática central |
|---|---|
| M1 | EDO do LIF + integração de Euler |
| M2 | STDP (exponenciais de timing), estabilidade de pesos |
| M3 | Sistemas dinâmicos acoplados (agente↔mundo) |
| M4 | Predictive coding, mínimos de energia, álgebra linear |
| M5 | Active inference, informação mútua, motivação intrínseca |
| M6 | Estatística experimental, análise de ablação |

> Toda fórmula aqui vira código testável nos marcos. Nada fica só no papel.
