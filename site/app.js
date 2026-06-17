/* brAIn: interações do site. Projeto de Rafael Sena Roman. */

(() => {
  "use strict";

  /* ===================== Dados dos marcos ===================== */
  const PHASES = [
    {
      sec: "§1", name: "As peças básicas",
      intro: "Cada peça do paradigma cerebral, construída do zero e testada contra a teoria e o acaso.",
      milestones: [
        { t: "Um neurônio vivo", d: "Um neurônio simulado que dispara como os de verdade. O comportamento bate com a teoria.", img: "m1_living_neuron.png" },
        { t: "A sinapse que aprende", d: "A conexão aprende por associação, como o cachorro de Pavlov. O grupo de controle não aprende, o que prova que é associação mesmo.", img: "m2_pavlov.png" },
        { t: "Corpo e mundo", d: "Posto num corpo simples, o agente chega à fonte de 'comida' em 98% das vezes, contra 28% do acaso.", img: "m3_embodiment.png" },
        { t: "Aprender a prever", d: "O agente aprende a antecipar o que vai sentir. A surpresa cai 79% e padrões surgem sozinhos.", img: "m4_predictive.png" },
        { t: "Curiosidade", d: "Curiosidade do tipo certo: busca o que dá para aprender e ignora barulho puro, que prende as versões ingênuas.", img: "m5_development.png" },
        { t: "O que é essencial", d: "Testes de remoção mostram o que importa: aprender, raciocinar sobre a entrada e ter capacidade. Acelerar na força bruta não resolve.", img: "m6_ablations.png" },
      ],
    },
    {
      sec: "§2", name: "O organismo",
      intro: "Juntar as peças num organismo só, que percebe, age, lembra, forma conceitos, planeja e enxerga.",
      milestones: [
        { t: "Tudo junto", d: "Corpo, previsão e curiosidade num agente só. Ele foge do barulho quatro vezes mais que o acaso.", img: "m7_integrated.png" },
        { t: "Memória", d: "Uma memória que repete experiências (como o sono faz) impede o agente de esquecer o que já aprendeu.", img: "m8_memory.png" },
        { t: "Conceitos", d: "Conceitos surgem: exemplos diferentes de uma mesma categoria viram quase o mesmo 'pensamento' no topo.", img: "m9_hierarchy.png" },
        { t: "Neurônios de verdade", d: "Tudo isso passa a rodar em neurônios que disparam de verdade, e ainda sem backpropagation.", img: "m10_spiking.png" },
        { t: "Pensar antes de agir", d: "O agente imagina futuros e planeja a rota. Resolve um labirinto que o agente reativo não consegue.", img: "m11_planning.png" },
        { t: "Visão", d: "Alimentado com imagens, ele desenvolve detectores de borda, parecidos com os do córtex visual humano.", img: "m12_vision.png" },
        { t: "A costura final", d: "Tudo reunido num organismo único. Só ele alcança o que está escondido atrás de uma parede.", img: "m13_integrated.png" },
      ],
    },
    {
      sec: "§3", name: "Os primeiros sinais de pensamento",
      intro: "Prever no tempo, aprender regras e inventar uma forma de se comunicar com sentido.",
      milestones: [
        { t: "Prever no tempo", d: "Aprende sequências ao longo do tempo e consegue imaginar como elas continuam.", img: "m14_temporal.png" },
        { t: "Gramática", d: "Aprende as regras de uma pequena 'gramática' e cria frases novas que respeitam essas regras.", img: "m15_grammar.png" },
        { t: "Uma língua emergente", d: "Dois agentes inventam, do zero, uma língua para se entenderem. Passam a acertar 100% das vezes.", img: "m16_communication.png" },
        { t: "Frases novas", d: "A língua deles funciona até para combinações que nunca tinham visto. É a raiz da linguagem.", img: "m17_compositional.png" },
        { t: "Palavras com sentido", d: "As palavras passam a significar o que o agente realmente percebe. Só dá para falar do que os dois conseguem distinguir.", img: "m18_grounded.png" },
        { t: "A língua nasce da vivência", d: "Cada agente aprende sozinho a reconhecer as coisas; depois os dois inventam uma língua sobre esses conceitos vividos, não sobre rótulos dados.", img: "m19_lived_language.png" },
        { t: "Um organismo vivo", d: "Num laço só, dois agentes percebem, lembram, são curiosos e falam ao mesmo tempo. Primeiro aprendem a ver; a linguagem vem logo atrás.", img: "m20_living_mind.png" },
        { t: "Palavras de português", d: "Como uma criança: vê o mundo e ouve a palavra certa junto. Aprende a dizer 'topo', 'centro', 'direita' do que vê, e a apontar quando ouve. Só nomeia o que consegue distinguir.", img: "m21_portuguese.png" },
        { t: "Eficiência como o cérebro", d: "O cérebro humano funciona com menos energia que uma lâmpada. Ativando poucos neurônios por vez, como no córtex, a rede faz o mesmo trabalho com cerca de 7 vezes menos operações. O caminho é o método, não a força bruta.", img: "m22_sparse_efficient.png" },
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
      const cap = `Figura ${figNo}. ${m.t}. ${m.d}`;
      return `
        <figure class="fig">
          <div class="fig__frame" data-img="site/img/${m.img}" data-cap="${cap}">
            <img loading="lazy" src="site/img/${m.img}" alt="${m.t}" />
          </div>
          <figcaption><span class="fig__num">Figura ${figNo}.</span>
            <span class="fig__title">${m.t}.</span> ${m.d}</figcaption>
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
  document.querySelectorAll("figure.fig, .plain, .abstract").forEach((el) => { el.classList.add("reveal"); io.observe(el); });

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
