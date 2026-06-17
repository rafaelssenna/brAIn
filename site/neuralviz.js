/* brAIn: o cérebro em atividade. Projeto de Rafael Sena Roman.
   Uma rede de neurônios LIF de verdade (modelo do M1) ligados por sinapses (M2):
   eles acumulam carga, disparam, e propagam pulsos que podem fazer outros
   dispararem. Ondas de atividade rodando ao vivo, como num cérebro. */

(() => {
  "use strict";
  const canvas = document.getElementById("neural-net");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let W = 0, H = 0, dpr = 1, neurons = [], edges = [], pulses = [];
  const N = 54;

  // neurônio LIF: tau=10, repouso/reset -65, limiar -50, R=10
  const TAU = 10, REST = -65, THRESH = -50, R = 10, REFR = 4;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    build();
  }

  function build() {
    neurons = [];
    const cx = W / 2, cy = H / 2, rx = W * 0.42, ry = H * 0.40;
    for (let i = 0; i < N; i++) {
      // distribuição orgânica dentro de uma elipse (aglomerado tipo-neural)
      const a = Math.random() * 6.283, r = Math.sqrt(Math.random());
      neurons.push({
        x: cx + Math.cos(a) * r * rx, y: cy + Math.sin(a) * r * ry,
        v: REST + Math.random() * 10, refr: 0, flash: 0, syn: 0, bg: 0.6 + Math.random() * 0.6,
      });
    }
    // sinapses: cada neurônio liga aos 3 vizinhos mais próximos (direcionado)
    edges = [];
    neurons.forEach((n, i) => { n.out = []; });
    for (let i = 0; i < N; i++) {
      const d = neurons.map((m, j) => [j, Math.hypot(m.x - neurons[i].x, m.y - neurons[i].y)])
        .filter(([j]) => j !== i).sort((a, b) => a[1] - b[1]);
      for (let k = 0; k < 3 && k < d.length; k++) {
        const j = d[k][0];
        edges.push([i, j]); neurons[i].out.push(j);
      }
    }
    pulses = [];
  }

  // estímulo pelo ponteiro
  const ptr = { x: -1, y: -1, on: false };
  function move(e) {
    const r = canvas.getBoundingClientRect(); const p = e.touches ? e.touches[0] : e;
    ptr.x = p.clientX - r.left; ptr.y = p.clientY - r.top; ptr.on = true;
  }
  canvas.addEventListener("mousemove", move);
  canvas.addEventListener("mouseleave", () => { ptr.on = false; });
  canvas.addEventListener("touchmove", (e) => { move(e); e.preventDefault(); }, { passive: false });

  const W_SYN = 0.85;      // peso sináptico (quanto um disparo empurra o alvo)
  const PSPEED = 0.07;     // velocidade do pulso pela sinapse

  function step() {
    for (const n of neurons) {
      let I = n.bg + n.syn; n.syn = 0;
      if (ptr.on) {                       // estímulo: corrente extra perto do cursor
        const d = Math.hypot(n.x - ptr.x, n.y - ptr.y);
        if (d < 80) I += (1 - d / 80) * 1.6;
      }
      if (n.refr > 0) { n.refr -= 1; n.v = REST; n.flash *= 0.8; continue; }
      n.v += (1 / TAU) * (-(n.v - REST) + R * I);
      if (n.v >= THRESH) {                // disparo
        n.v = REST; n.refr = REFR; n.flash = 1;
        for (const j of n.out) pulses.push({ a: n, b: neurons[j], t: 0 });
      }
      n.flash *= 0.82;
    }
    for (let k = pulses.length - 1; k >= 0; k--) {
      const p = pulses[k]; p.t += PSPEED;
      if (p.t >= 1) { p.b.syn += W_SYN; pulses.splice(k, 1); }
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    // sinapses
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(138,180,248,0.06)";
    ctx.beginPath();
    for (const [i, j] of edges) { ctx.moveTo(neurons[i].x, neurons[i].y); ctx.lineTo(neurons[j].x, neurons[j].y); }
    ctx.stroke();
    // pulsos viajando (potenciais de ação)
    for (const p of pulses) {
      const x = p.a.x + (p.b.x - p.a.x) * p.t, y = p.a.y + (p.b.y - p.a.y) * p.t;
      ctx.fillStyle = "rgba(160,200,255,0.9)";
      ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 6.283); ctx.fill();
    }
    // neurônios
    for (const n of neurons) {
      const exc = Math.max(0, (n.v - REST) / (THRESH - REST));   // 0..1 carga
      const lvl = Math.min(1, exc * 0.5 + n.flash);
      if (n.flash > 0.05) {                                       // brilho ao disparar
        ctx.fillStyle = `rgba(190,215,255,${0.18 * n.flash})`;
        ctx.beginPath(); ctx.arc(n.x, n.y, 9 + 7 * n.flash, 0, 6.283); ctx.fill();
      }
      ctx.fillStyle = `rgba(${170 + 70 * lvl},${190 + 55 * lvl},${230 + 25 * lvl},${0.45 + 0.55 * lvl})`;
      ctx.beginPath(); ctx.arc(n.x, n.y, 2.4 + 2.2 * lvl, 0, 6.283); ctx.fill();
    }
  }

  function loop() { step(); draw(); requestAnimationFrame(loop); }
  resize();
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(resize, 150); });
  loop();
})();
