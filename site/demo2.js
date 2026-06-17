/* brAIn: demo do estágio atual. Projeto de Rafael Sena Roman.
   Dois agentes inventam uma língua compartilhada (mecanismo do M16): o falante
   nomeia um conceito, o ouvinte adivinha; acertando, reforçam. Aprendizado por
   recompensa + atenção conjunta. Roda ao vivo no navegador. */

(() => {
  "use strict";
  const exEl = document.getElementById("ln-ex");
  const lexEl = document.getElementById("ln-lex");
  const accEl = document.getElementById("ln-acc");
  const roundEl = document.getElementById("ln-round");
  const resetBtn = document.getElementById("ln-reset");
  if (!exEl || !lexEl) return;

  const N = 5;                                   // conceitos
  const WORDS = ["ka", "wo", "ti", "lu", "ne"];  // "palavras" inventadas
  const COLORS = ["#e07a5f", "#5b8def", "#5cb88a", "#e3b341", "#9b6ad6"];
  const LR = 0.18, TEMP = 0.4;

  let S, R, round, cells, timer, done;

  function noise() { return (Math.random() - 0.5) * 0.1; }
  function softmax(v) {
    const mx = Math.max(...v), ex = v.map((x) => Math.exp((x - mx) / TEMP));
    const s = ex.reduce((a, b) => a + b, 0); return ex.map((x) => x / s);
  }
  function argmax(v) { let bi = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[bi]) bi = i; return bi; }
  function sample(p) { let r = Math.random(), a = 0; for (let i = 0; i < p.length; i++) { a += p[i]; if (r <= a) return i; } return p.length - 1; }

  function buildGrid() {
    lexEl.style.gridTemplateColumns = `24px repeat(${N}, 1fr)`;
    lexEl.innerHTML = "";
    lexEl.appendChild(document.createElement("div"));         // canto
    WORDS.forEach((w) => { const h = document.createElement("div"); h.className = "lex-wordh"; h.textContent = w; lexEl.appendChild(h); });
    cells = [];
    for (let c = 0; c < N; c++) {
      const dot = document.createElement("div"); dot.className = "lex-dot"; dot.style.background = COLORS[c]; lexEl.appendChild(dot);
      cells[c] = [];
      for (let m = 0; m < N; m++) {
        const cell = document.createElement("div"); cell.className = "lex-cell"; lexEl.appendChild(cell); cells[c][m] = cell;
      }
    }
  }

  function init() {
    S = Array.from({ length: N }, () => Array.from({ length: N }, noise));
    R = Array.from({ length: N }, () => Array.from({ length: N }, noise));
    round = 0; done = false;
    buildGrid(); updateGrid();
    accEl.textContent = Math.round(100 / N) + "%"; roundEl.textContent = "0";
    exEl.innerHTML = '<span class="muted">Eles ainda não têm vocabulário. Observe a língua nascer…</span>';
  }

  function playRound() {
    const c = Math.floor(Math.random() * N);
    const m = sample(softmax(S[c]));
    const g = argmax(R[m]);
    const reward = g === c;
    R[m][c] += LR;                                // atenção conjunta (vê o alvo real)
    S[c][m] += reward ? LR : -LR;                 // recompensa
    round++;
    return { c, m, g, reward };
  }

  function evalAcc() {
    let ok = 0;
    for (let c = 0; c < N; c++) ok += (argmax(R[argmax(S[c])]) === c) ? 1 : 0;
    return ok / N;
  }

  function updateGrid() {
    for (let c = 0; c < N; c++) {
      const p = softmax(S[c]);
      for (let m = 0; m < N; m++) cells[c][m].style.background = `rgba(138,180,248,${(p[m] * 0.95).toFixed(3)})`;
    }
  }

  function showExchange(r) {
    const dot = (i) => `<span class="swatch" style="background:${COLORS[i]}"></span>`;
    exEl.innerHTML =
      `<span class="muted">Falante vê</span> ${dot(r.c)} <span class="muted">e diz</span> ` +
      `<span class="word">${WORDS[r.m]}</span><span class="muted">. Ouvinte entende</span> ${dot(r.g)} ` +
      (r.reward ? '<span class="ok">✓</span>' : '<span class="no">✗</span>');
  }

  function tick() {
    let last;
    for (let i = 0; i < 2; i++) last = playRound();   // 2 rodadas por passo
    showExchange(last); updateGrid();
    const acc = evalAcc();
    accEl.textContent = Math.round(acc * 100) + "%"; roundEl.textContent = round;
    if (acc >= 1) {
      done = true; clearInterval(timer);
      exEl.innerHTML = '<span class="ok">✓</span> <span>Língua formada: agora eles se entendem 100% das vezes, com um vocabulário que inventaram do zero.</span>';
    }
  }

  function start() { clearInterval(timer); init(); timer = setInterval(tick, 280); }
  if (resetBtn) resetBtn.addEventListener("click", start);

  // só roda quando a seção entra na tela (economiza e dá o efeito de "começar do zero")
  const io = new IntersectionObserver((es) => {
    es.forEach((e) => { if (e.isIntersecting && !timer) start(); });
  }, { threshold: 0.3 });
  io.observe(document.getElementById("agora"));
})();
