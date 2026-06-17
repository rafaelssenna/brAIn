/* brAIn — relato de projeto · interações
   Projeto de Rafael Sena Roman; implementação assistida por IA. */

(() => {
  "use strict";

  /* ===================== Dados dos marcos ===================== */
  const PHASES = [
    {
      sec: "§1", name: "Os ingredientes",
      intro: "Cada peça do paradigma cerebral, implementada do zero e validada contra teoria e baseline.",
      milestones: [
        { n: "M1", t: "Neurônio vivo", d: "Um neurônio LIF que dispara como os reais — a simulação bate com a curva f–I analítica.", img: "m1_living_neuron.png" },
        { n: "M2", t: "Sinapse que aprende", d: "STDP e condicionamento pavloviano; o controle não-pareado não aprende — logo é associativo.", img: "m2_pavlov.png" },
        { n: "M3", t: "Corpo e mundo", d: "Fototaxia: chega à fonte em 98% dos episódios vs 28% do acaso. O laço percepção–ação fecha.", img: "m3_embodiment.png" },
        { n: "M4", t: "Máquina de previsão", d: "Predictive coding: surpresa −79%; campos receptivos emergem; regra local, sem backprop.", img: "m4_predictive.png" },
        { n: "M5", t: "Curiosidade", d: "Learning progress: foge da 'TV chiando'. A novelty-seeking colapsa (98% presa no ruído).", img: "m5_development.png" },
        { n: "M6", t: "Ablação e escala", d: "Aprendizado, inferência e capacidade são essenciais; vetorizar ingênuo não escala (resultado negativo honesto).", img: "m6_ablations.png" },
      ],
    },
    {
      sec: "§2", name: "O organismo integrado",
      intro: "Costurar as peças num único ser que percebe, age, lembra, conceitua, planeja e vê.",
      milestones: [
        { n: "M7", t: "Laço integrado", d: "Corpo, previsão e curiosidade num agente só. Evita fisicamente o ruído 4×.", img: "m7_integrated.png" },
        { n: "M8", t: "Memória / replay", d: "O replay vence o esquecimento (+0.44 → +0.004) e destrava a curiosidade (93%→99%).", img: "m8_memory.png" },
        { n: "M9", t: "Hierarquia", d: "Conceitos emergem: a invariância de categoria sobe de 0.81 a 0.99 na camada do topo.", img: "m9_hierarchy.png" },
        { n: "M10", t: "Substrato spiking", d: "Predictive coding em neurônios LIF; surpresa −82%. A fratura rate-vs-spiking, costurada.", img: "m10_spiking.png" },
        { n: "M11", t: "Agência (planejar)", d: "Imagina futuros e planeja a rota: 100% vs 83% reativo. O primeiro 'pensar à frente'.", img: "m11_planning.png" },
        { n: "M12", t: "Visão", d: "Detectores de borda orientados emergem da entrada visual — como no córtex visual V1.", img: "m12_vision.png" },
        { n: "M13", t: "A costura final", d: "Tudo num laço só. Apenas o organismo integrado alcança o conteúdo atrás da parede (10.7% vs 0%).", img: "m13_integrated.png" },
      ],
    },
    {
      sec: "§3", name: "A cognição",
      intro: "Prever no tempo, internalizar regras, e inventar uma língua ancorada na percepção.",
      milestones: [
        { n: "M14", t: "Previsão temporal", d: "Aprende sequências no tempo e imagina a continuação correta — o substrato do pensar.", img: "m14_temporal.png" },
        { n: "M15", t: "Gramática", d: "Aprende REGRAS (não uma sequência fixa) e gera frases novas 98% gramaticais.", img: "m15_grammar.png" },
        { n: "M16", t: "Comunicação emergente", d: "Dois agentes inventam uma língua compartilhada do zero: do acaso a 100% de acerto.", img: "m16_communication.png" },
        { n: "M17", t: "Composicionalidade", d: "O código composicional generaliza para combinações nunca vistas (100% vs 0% holístico).", img: "m17_compositional.png" },
        { n: "M18", t: "Grounding", d: "Símbolos ancorados na percepção (Harnad). Só se comunica o que ambos distinguem (r=0.90).", img: "m18_grounded.png" },
      ],
    },
  ];

  /* ===================== Render das figuras ===================== */
  let figNo = 0;
  PHASES.forEach((ph, i) => {
    const host = document.getElementById(`fase-${i + 1}`);
    if (!host) return;
    const figs = ph.milestones.map((m) => {
      figNo += 1;
      const cap = `${m.n} — ${m.t}: ${m.d}`;
      return `
        <figure class="fig reveal">
          <div class="fig__frame" data-img="site/img/${m.img}" data-cap="Figura ${figNo}. ${cap}">
            <img loading="lazy" src="site/img/${m.img}" alt="${m.n} — ${m.t}" />
          </div>
          <figcaption><span class="fig__num">Figura ${figNo}.</span>
            <span class="fig__title">${m.n} — ${m.t}.</span> ${m.d}</figcaption>
        </figure>`;
    }).join("");
    host.innerHTML = `
      <h2 class="h2"><span class="sec-no">${ph.sec}</span> ${ph.name}</h2>
      <p class="phase-intro">${ph.intro}</p>
      <div class="figgrid">${figs}</div>`;
  });

  /* ===================== Reveal sutil ===================== */
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("is-visible"); io.unobserve(e.target); } });
  }, { threshold: 0.1 });
  document.querySelectorAll(".reveal, figure.fig").forEach((el) => { el.classList.add("reveal"); io.observe(el); });

  /* ===================== Barra de leitura + topo ===================== */
  const bar = document.getElementById("reading-bar");
  const topbar = document.getElementById("topbar");
  const onScroll = () => {
    const st = window.scrollY;
    const h = document.documentElement.scrollHeight - window.innerHeight;
    if (bar) bar.style.width = (h > 0 ? (st / h) * 100 : 0) + "%";
    if (topbar) topbar.classList.toggle("is-scrolled", st > 30);
  };
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  /* ===================== Lightbox ===================== */
  const lb = document.getElementById("lightbox");
  const lbImg = document.getElementById("lightbox-img");
  const lbCap = document.getElementById("lightbox-cap");
  const close = () => { lb.classList.remove("is-open"); lb.setAttribute("aria-hidden", "true"); lbImg.src = ""; };
  document.addEventListener("click", (e) => {
    const frame = e.target.closest(".fig__frame");
    if (frame) { lbImg.src = frame.dataset.img; lbImg.alt = frame.dataset.cap; lbCap.textContent = frame.dataset.cap; lb.classList.add("is-open"); lb.setAttribute("aria-hidden", "false"); return; }
    if (e.target.closest(".lightbox__close") || e.target === lb) close();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
})();
