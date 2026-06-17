/* brAIn — portfólio científico · interações
   Projeto de Rafael S. Senna; implementação assistida por IA. */

(() => {
  "use strict";

  /* ===================== Dados dos marcos ===================== */
  const PHASES = [
    {
      idx: "FASE 1",
      name: "Os ingredientes",
      desc: "Cada peça do paradigma cerebral, implementada do zero e validada contra teoria e baseline.",
      milestones: [
        { n: "M1", t: "Neurônio vivo", d: "Um neurônio LIF que dispara como os reais — a simulação bate com a curva f–I analítica.", img: "m1_living_neuron.png" },
        { n: "M2", t: "Sinapse que aprende", d: "STDP: condicionamento pavloviano. O controle não-pareado não aprende — logo é associativo.", img: "m2_pavlov.png" },
        { n: "M3", t: "Corpo e mundo", d: "Fototaxia 98% vs 28% (acaso). O laço percepção→ação→percepção se fecha.", img: "m3_embodiment.png" },
        { n: "M4", t: "Máquina de previsão", d: "Predictive coding: surpresa −79%, campos receptivos emergem, regra local, sem backprop.", img: "m4_predictive.png" },
        { n: "M5", t: "Curiosidade", d: "Learning progress: foge da 'TV chiando'. A novelty-seeking colapsa (98% presa no ruído).", img: "m5_development.png" },
        { n: "M6", t: "Ablação + escala", d: "Aprendizado/inferência/capacidade são essenciais; vetorizar ingênuo não escala (resultado negativo honesto).", img: "m6_ablations.png" },
      ],
    },
    {
      idx: "FASE 2",
      name: "O organismo integrado",
      desc: "Costurar as peças num único ser que percebe, age, lembra, conceitua, planeja e vê.",
      milestones: [
        { n: "M7", t: "Laço integrado", d: "Corpo + previsão + curiosidade num agente só. Evita fisicamente o ruído 4×.", img: "m7_integrated.png" },
        { n: "M8", t: "Memória / replay", d: "O replay vence o esquecimento (+0.44 → +0.004) e destrava a curiosidade (93%→99%).", img: "m8_memory.png" },
        { n: "M9", t: "Hierarquia", d: "Conceitos emergem: invariância de categoria sobe de 0.81 a 0.99 no topo.", img: "m9_hierarchy.png" },
        { n: "M10", t: "Substrato spiking", d: "Predictive coding em neurônios LIF; surpresa −82%. A fratura rate-vs-spiking, costurada.", img: "m10_spiking.png" },
        { n: "M11", t: "Agência (pensar)", d: "Imagina futuros e planeja a rota: 100% vs 83% reativo. O primeiro 'pensar à frente'.", img: "m11_planning.png" },
        { n: "M12", t: "Visão", d: "Detectores de borda orientados emergem da entrada visual — como o córtex V1.", img: "m12_vision.png" },
        { n: "M13", t: "A costura final", d: "Tudo num laço só. Só o organismo integrado alcança o conteúdo atrás da parede (10.7% vs 0%).", img: "m13_integrated.png" },
      ],
    },
    {
      idx: "FASE 3",
      name: "A cognição",
      desc: "Prever no tempo, internalizar regras, e inventar uma língua ancorada na percepção.",
      milestones: [
        { n: "M14", t: "Previsão temporal", d: "Aprende sequências no tempo e imagina a continuação correta — o substrato do pensar.", img: "m14_temporal.png" },
        { n: "M15", t: "Gramática", d: "Aprende REGRAS (não uma sequência fixa) e gera frases novas 98% gramaticais.", img: "m15_grammar.png" },
        { n: "M16", t: "Comunicação emergente", d: "Dois agentes inventam uma língua compartilhada do zero: acaso → 100%.", img: "m16_communication.png" },
        { n: "M17", t: "Composicionalidade", d: "Código composicional generaliza para combinações nunca vistas (100% vs 0% holístico).", img: "m17_compositional.png" },
        { n: "M18", t: "Grounding", d: "Símbolos ancorados na percepção (Harnad). Só se comunica o que ambos distinguem (r=0.90).", img: "m18_grounded.png" },
      ],
    },
  ];

  /* ===================== Render dos marcos ===================== */
  const phasesEl = document.getElementById("phases");
  if (phasesEl) {
    PHASES.forEach((ph) => {
      const sec = document.createElement("div");
      sec.className = "phase";
      const cards = ph.milestones.map((m) => `
        <article class="mcard reveal">
          <div class="mcard__imgwrap" data-img="site/img/${m.img}" data-cap="${m.n} — ${m.t}: ${m.d}">
            <img class="mcard__img" loading="lazy" src="site/img/${m.img}" alt="${m.n} — ${m.t}" />
            <span class="mcard__badge">${m.n}</span>
          </div>
          <div class="mcard__body">
            <h4 class="mcard__title">${m.t}</h4>
            <p class="mcard__disc">${m.d}</p>
          </div>
        </article>`).join("");
      sec.innerHTML = `
        <div class="phase__head reveal">
          <span class="phase__idx">${ph.idx}</span>
          <span class="phase__name">${ph.name}</span>
        </div>
        <p class="phase__desc reveal">${ph.desc}</p>
        <div class="grid">${cards}</div>`;
      phasesEl.appendChild(sec);
    });
  }

  /* ===================== Reveal no scroll ===================== */
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("is-visible"); io.unobserve(e.target); } });
  }, { threshold: 0.12 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

  /* ===================== Nav: estado ao rolar ===================== */
  const nav = document.getElementById("nav");
  const onScroll = () => nav.classList.toggle("is-scrolled", window.scrollY > 40);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  /* ===================== Contadores das stats ===================== */
  const statsEl = document.getElementById("stats");
  if (statsEl) {
    const animateStats = () => {
      statsEl.querySelectorAll(".stat__num").forEach((el) => {
        const target = +el.dataset.target;
        if (target === 0) { el.textContent = "0"; return; }
        const dur = 1100; const t0 = performance.now();
        const tick = (t) => {
          const p = Math.min((t - t0) / dur, 1);
          el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3)));
          if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      });
    };
    const so = new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting) { animateStats(); so.disconnect(); } });
    }, { threshold: 0.5 });
    so.observe(statsEl);
  }

  /* ===================== Lightbox ===================== */
  const lb = document.getElementById("lightbox");
  const lbImg = document.getElementById("lightbox-img");
  const lbCap = document.getElementById("lightbox-cap");
  const openLb = (src, cap) => { lbImg.src = src; lbImg.alt = cap; lbCap.textContent = cap; lb.classList.add("is-open"); lb.setAttribute("aria-hidden", "false"); };
  const closeLb = () => { lb.classList.remove("is-open"); lb.setAttribute("aria-hidden", "true"); lbImg.src = ""; };
  document.addEventListener("click", (e) => {
    const wrap = e.target.closest(".mcard__imgwrap");
    if (wrap) { openLb(wrap.dataset.img, wrap.dataset.cap); return; }
    if (e.target.closest(".lightbox__close") || e.target === lb) closeLb();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeLb(); });

  /* ===================== Animação de rede neural ===================== */
  const canvas = document.getElementById("neural-bg");
  if (canvas && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const ctx = canvas.getContext("2d");
    let w, h, dpr, nodes, pulses;

    const count = () => {
      const a = window.innerWidth * window.innerHeight;
      return Math.max(28, Math.min(90, Math.round(a / 22000)));
    };

    function init() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.width = window.innerWidth * dpr;
      h = canvas.height = window.innerHeight * dpr;
      canvas.style.width = window.innerWidth + "px";
      canvas.style.height = window.innerHeight + "px";
      const n = count();
      nodes = Array.from({ length: n }, () => ({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.18 * dpr, vy: (Math.random() - 0.5) * 0.18 * dpr,
        r: (Math.random() * 1.6 + 1.0) * dpr,
      }));
      pulses = [];
    }

    const LINK = 150;
    function frame() {
      ctx.clearRect(0, 0, w, h);
      const link = LINK * dpr;
      // mover
      for (const p of nodes) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
      }
      // conexões
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const dist = Math.hypot(dx, dy);
          if (dist < link) {
            const o = (1 - dist / link) * 0.5;
            ctx.strokeStyle = `rgba(120,160,235,${o})`;
            ctx.lineWidth = dpr * 0.6;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
            // disparo ocasional ao longo da conexão
            if (Math.random() < 0.0006) pulses.push({ a, b, t: 0 });
          }
        }
      }
      // nós
      for (const p of nodes) {
        ctx.fillStyle = "rgba(34,211,238,0.85)";
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
      }
      // pulsos (spikes viajando)
      for (let k = pulses.length - 1; k >= 0; k--) {
        const pu = pulses[k]; pu.t += 0.03;
        if (pu.t >= 1) { pulses.splice(k, 1); continue; }
        const x = pu.a.x + (pu.b.x - pu.a.x) * pu.t;
        const y = pu.a.y + (pu.b.y - pu.a.y) * pu.t;
        ctx.fillStyle = "rgba(167,139,250,0.95)";
        ctx.beginPath(); ctx.arc(x, y, dpr * 2.2, 0, Math.PI * 2); ctx.fill();
      }
      requestAnimationFrame(frame);
    }

    init();
    frame();
    let rt;
    window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(init, 200); });
  }
})();
