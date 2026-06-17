/* brAIn: demo viva. Projeto de Rafael Sena Roman.
   Uma criatura com neurônios LIF reais (modelo do M1) e fiação cruzada (M3) que
   persegue a luz. Rodando ao vivo no navegador. */

(() => {
  "use strict";
  const canvas = document.getElementById("brain-demo");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let W = 0, H = 0, dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* --- luz: segue o ponteiro; sem interação, vagueia sozinha --- */
  const light = { x: 0, y: 0, autoT: 0, lastInput: -1e9 };
  function pointer(e) {
    const r = canvas.getBoundingClientRect();
    const p = e.touches ? e.touches[0] : e;
    light.x = p.clientX - r.left; light.y = p.clientY - r.top;
    light.lastInput = performance.now();
  }
  canvas.addEventListener("mousemove", pointer);
  canvas.addEventListener("touchstart", (e) => { pointer(e); }, { passive: true });
  canvas.addEventListener("touchmove", (e) => { pointer(e); e.preventDefault(); }, { passive: false });

  /* --- neurônio LIF (o mesmo do M1): integra corrente, dispara, reseta --- */
  function lif(s, I) {
    if (s.refr > 0) { s.refr -= 1; s.v = -65; return 0; }
    s.v += (1 / 10) * (-(s.v + 65) + 10 * I);     // tau=10, R=10, repouso -65
    if (s.v >= -50) { s.v = -65; s.refr = 2; return 1; }   // limiar -50, refratário 2
    return 0;
  }

  /* --- a criatura --- */
  const c = { x: 0, y: 0, h: 0, eyeL: { v: -65, refr: 0 }, eyeR: { v: -65, refr: 0 }, sL: 0, sR: 0, trail: [] };
  function reset() { c.x = W / 2; c.y = H / 2; c.h = Math.random() * 6.28; c.trail = []; }

  const WIN = 14;          // janela de simulação por quadro (ms)
  const EYE_OFF = 0.5;     // ângulo dos olhos vs. frente
  const I_BIAS = 1.0, I_GAIN = 4.0;          // sensor -> corrente
  const VMIN_L = 0.35, VMIN_R = 0.7, K = 2.4, WB = 16, STEP = 1.35;  // motores/rodas

  function senseRate(neuron, I) {
    let n = 0;
    for (let i = 0; i < WIN; i++) n += lif(neuron, I);
    return Math.min(n / 5, 1);     // taxa normalizada (0..1)
  }

  function update() {
    const fall = Math.min(W, H) * 0.5;
    // luz automática quando ninguém mexe
    if (performance.now() - light.lastInput > 2200) {
      light.autoT += 0.012;
      light.x = W * (0.5 + 0.36 * Math.cos(light.autoT));
      light.y = H * (0.5 + 0.32 * Math.sin(light.autoT * 1.4));
    }
    const dx = light.x - c.x, dy = light.y - c.y;
    const dist = Math.hypot(dx, dy);
    const ang = Math.atan2(dy, dx);
    const inten = 1 / (1 + (dist / fall) * (dist / fall));
    const actL = inten * Math.max(0, Math.cos(ang - (c.h + EYE_OFF)));
    const actR = inten * Math.max(0, Math.cos(ang - (c.h - EYE_OFF)));
    c.sL = senseRate(c.eyeL, I_BIAS + I_GAIN * actL);
    c.sR = senseRate(c.eyeR, I_BIAS + I_GAIN * actR);
    // fiação CRUZADA: olho esquerdo -> roda direita; olho direito -> roda esquerda
    const vLeft = VMIN_L + K * c.sR;
    const vRight = VMIN_R + K * c.sL;
    const fwd = 0.5 * (vLeft + vRight) * STEP;
    c.h += ((vRight - vLeft) / WB) * STEP;
    c.x += fwd * Math.cos(c.h); c.y += fwd * Math.sin(c.h);
    // quica nas bordas
    if (c.x < 14) { c.x = 14; c.h = Math.PI - c.h; }
    if (c.x > W - 14) { c.x = W - 14; c.h = Math.PI - c.h; }
    if (c.y < 14) { c.y = 14; c.h = -c.h; }
    if (c.y > H - 14) { c.y = H - 14; c.h = -c.h; }
    c.trail.push([c.x, c.y]); if (c.trail.length > 60) c.trail.shift();
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    // luz (brilho quente)
    const g = ctx.createRadialGradient(light.x, light.y, 0, light.x, light.y, 70);
    g.addColorStop(0, "rgba(255,244,214,0.95)");
    g.addColorStop(0.4, "rgba(255,210,120,0.35)");
    g.addColorStop(1, "rgba(255,210,120,0)");
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#fff6e0"; ctx.beginPath(); ctx.arc(light.x, light.y, 5, 0, 6.29); ctx.fill();
    // rastro
    ctx.strokeStyle = "rgba(138,180,248,0.20)"; ctx.lineWidth = 2; ctx.beginPath();
    c.trail.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke();
    // corpo
    ctx.save(); ctx.translate(c.x, c.y); ctx.rotate(c.h);
    ctx.fillStyle = "#c9d2e0"; ctx.beginPath();
    ctx.moveTo(12, 0); ctx.lineTo(-8, 7); ctx.lineTo(-8, -7); ctx.closePath(); ctx.fill();
    // olhos: acendem (âmbar) conforme disparam
    eye(6, -5, c.sR); eye(6, 5, c.sL);
    ctx.restore();
  }
  function eye(x, y, rate) {
    ctx.fillStyle = `rgba(255,200,90,${0.25 + 0.75 * rate})`;
    ctx.beginPath(); ctx.arc(x, y, 2.4 + 2.4 * rate, 0, 6.29); ctx.fill();
  }

  function loop() { update(); draw(); requestAnimationFrame(loop); }
  resize(); reset();
  light.x = W * 0.8; light.y = H * 0.3;
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { resize(); reset(); }, 150); });
  loop();
})();
