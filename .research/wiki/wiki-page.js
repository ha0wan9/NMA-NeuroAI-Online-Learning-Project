(function () {
  "use strict";

  const registry = window.RESEARCH_WIKI;
  if (!registry) return;

  const LATEX = {
    "continual-learning": String.raw`\mathcal{D}_1,\mathcal{D}_2,\ldots,\mathcal{D}_T,\quad \mathcal{D}_t\neq\mathcal{D}_{t+1},\qquad \theta_t=\mathcal{A}(\theta_{t-1},\mathcal{D}_t)`,
    "online-learning": String.raw`\hat y_t=f_{\theta_t}(x_t),\qquad \ell_t=\ell(\hat y_t,y_t),\qquad \theta_{t+1}=\mathcal{U}(\theta_t,x_t,y_t)`,
    "static-training": String.raw`\mathcal{R}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\!\left[\ell\!\left(f_\theta(x),y\right)\right]`,
    "class-incremental-learning": String.raw`\mathcal{C}_{\leq t}=\bigcup_{k=1}^{t}\mathcal{C}_k,\qquad \hat y=\underset{c\in\mathcal{C}_{\leq t}}{\arg\max}\; f_\theta(x)_c`,
    "domain-incremental-learning": String.raw`\mathcal{Y}_t=\mathcal{Y}\ \ \forall t,\qquad p_t(x\mid y)\neq p_{t+1}(x\mid y)`,
    "task-id": String.raw`\text{task-aware: }\hat y=f_\theta(x,t),\qquad \text{task-agnostic: }\hat y=f_\theta(x)`,
    "shared-head": String.raw`z=f_\theta(x)\in\mathbb{R}^{|\mathcal C|},\qquad \hat y=\underset{c}{\arg\max}\;z_c`,
    "replay": String.raw`\mathcal B_t=\mathcal B_t^{\mathrm{current}}\cup\mathcal B_t^{\mathrm{replay}},\qquad \mathcal B_t^{\mathrm{replay}}\subset\bigcup_{k<t}\mathcal D_k`,
    "single-pass": String.raw`\operatorname{count}_{\mathrm{train}}(e_i)=1\qquad \forall e_i\in\mathcal S`,
    "cifar-10": String.raw`\mathcal D_{\mathrm{CIFAR\text{-}10}}\subset\mathbb R^{32\times32\times3}\times\{1,\ldots,10\},\qquad |\mathcal D|=60{,}000`,
    "mnist": String.raw`\mathcal D_{\mathrm{MNIST}}\subset[0,255]^{28\times28}\times\{0,\ldots,9\},\qquad |\mathcal D_{\mathrm{train}}|=60{,}000`,
    "split-mnist": String.raw`\mathcal C_1=\{0,1\},\ \mathcal C_2=\{2,3\},\ \mathcal C_3=\{4,5\},\ \mathcal C_4=\{6,7\},\ \mathcal C_5=\{8,9\}`,
    "permuted-mnist": String.raw`x^{(t)}_j=x_{\pi_t(j)},\qquad \pi_t:\{1,\ldots,784\}\rightarrow\{1,\ldots,784\}\ \text{is bijective}`,
    "split-cifar-10": String.raw`\{1,\ldots,10\}=\mathcal C_1\,\dot\cup\,\mathcal C_2\,\dot\cup\,\mathcal C_3\,\dot\cup\,\mathcal C_4\,\dot\cup\,\mathcal C_5,\qquad |\mathcal C_t|=2`,
    "prequential-evaluation": String.raw`\operatorname{Acc}_{\mathrm{preq}}=\frac{1}{N}\sum_{t=1}^{N}\mathbf 1\!\left[\underset{c}{\arg\max}\;f_{\theta_t}(x_t)_c=y_t\right]`,
    "accuracy": String.raw`\operatorname{Acc}(\theta;\mathcal D)=\frac{1}{|\mathcal D|}\sum_{(x,y)\in\mathcal D}\mathbf 1\!\left[\underset{c}{\arg\max}\;f_\theta(x)_c=y\right]`,
    "final-average-accuracy": String.raw`\operatorname{ACC}_{\mathrm{final}}=\frac{1}{T}\sum_{i=1}^{T}R_{T,i}`,
    "adaptation-gain": String.raw`A_i=R_{i+1,i}-R_{i,i},\qquad \bar A=\frac{1}{T}\sum_{i=1}^{T}A_i`,
    "average-forgetting": String.raw`F=\frac{1}{T-1}\sum_{i=1}^{T-1}\left(\max_{t\ge i+1}R_{t,i}-R_{T,i}\right)`,
    "backward-transfer": String.raw`\operatorname{BWT}=\frac{1}{T-1}\sum_{i=1}^{T-1}\left(R_{T,i}-R_{i+1,i}\right)`,
    "catastrophic-forgetting": String.raw`R_{i+1,i}\ \text{is high},\qquad R_{T,i}\ll R_{i+1,i}`,
    "stability-plasticity-trade-off": String.raw`\underset{\theta}{\operatorname{minimize}}\;F(\theta)\quad\text{subject to}\quad A(\theta)\ge A_{\min},\ \ P(\theta)\ge P_{\min}`,
    "percentage-point": String.raw`\Delta_{\mathrm{pp}}=100(a-b),\qquad \Delta_{\mathrm{relative}}=\frac{a-b}{b}`,
    "backpropagation": String.raw`\delta^{L}=\frac{\partial\ell}{\partial z^{L}},\qquad \delta^{l}=(W^{l+1})^{\!\top}\delta^{l+1}\odot\phi'(z^l),\qquad \frac{\partial\ell}{\partial W^l}=\delta^l(a^{l-1})^{\!\top}`,
    "repeated-update-backpropagation": String.raw`\theta_{k+1}=\operatorname{Adam}\!\left(\theta_k,\nabla_\theta\ell(\mathcal B;\theta_k)\right),\qquad k=0,\ldots,K-1,\ \ K=16`,
    "predictive-coding": String.raw`E(v,\theta)=\sum_l\frac12\left\|v_l-f_l(v_{l-1};\theta_l)\right\|_2^2,\qquad v\leftarrow v-\eta_v\nabla_vE,\qquad \theta\leftarrow\theta-\eta_\theta\nabla_\theta E`,
    "relaxation": String.raw`v^{(\tau+1)}=v^{(\tau)}-\eta_v\nabla_vE\!\left(v^{(\tau)},\theta\right),\qquad \tau=0,\ldots,T-1`,
    "stochastic-gradient-descent": String.raw`g_t=\frac{1}{|\mathcal B_t|}\sum_{i\in\mathcal B_t}\nabla_\theta\ell_i(\theta_t),\qquad \theta_{t+1}=\theta_t-\eta_tg_t`,
    "adam": String.raw`\begin{aligned}m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\\\theta_t&=\theta_{t-1}-\eta\,\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}.\end{aligned}`,
    "learning-rate": String.raw`\theta_{t+1}=\theta_t-\eta_t d_t`,
    "weight-decay": String.raw`\mathcal L_{\mathrm{reg}}(\theta)=\mathcal L(\theta)+\frac{\lambda}{2}\|\theta\|_2^2,\qquad \theta\leftarrow(1-\eta\lambda)\theta-\eta\nabla\mathcal L`,
    "half-squared-error": String.raw`\ell(z,y)=\frac12\|z-y\|_2^2,\qquad \frac{\partial\ell}{\partial z}=z-y`,
    "logits": String.raw`\hat y=\underset{k}{\arg\max}\;z_k,\qquad \operatorname{softmax}(z)_k=\frac{e^{z_k}}{\sum_j e^{z_j}}`,
    "update-alignment": String.raw`\operatorname{align}(\Delta\theta^{a},\Delta\theta^{b})=\frac{(\Delta\theta^{a})^{\!\top}\Delta\theta^{b}}{\|\Delta\theta^{a}\|_2\,\|\Delta\theta^{b}\|_2}`,
    "cosine-similarity": String.raw`\cos(a,b)=\frac{a^\top b}{\|a\|_2\,\|b\|_2}\in[-1,1]`,
    "signal-to-noise-ratio": String.raw`\operatorname{SNR}=\frac{\|\mathbb E[g]\|_2}{\sqrt{\mathbb E\!\left[\|g-\mathbb E[g]\|_2^2\right]}}`,
    "representation": String.raw`h_l:\mathcal X\rightarrow\mathbb R^{d_l},\qquad H_l=\begin{bmatrix}h_l(x_1)^\top\\\vdots\\h_l(x_n)^\top\end{bmatrix}`,
    "convolutional-layer": String.raw`h_{k,i,j}=\phi\!\left(b_k+\sum_c\sum_{u,v}W_{k,c,u,v}\,x_{c,i+u,j+v}\right)`,
    "fully-connected-layer": String.raw`h=\phi(Wx+b)`,
    "rdm": String.raw`\operatorname{RDM}_{ij}=1-\operatorname{corr}\!\left(h(x_i),h(x_j)\right),\qquad \operatorname{RDM}_{ii}=0`,
    "representation-drift": String.raw`\operatorname{drift}(R^{0},R^{t})=1-\operatorname{corr}\!\left(\operatorname{vec}_{\triangle}(R^{0}),\operatorname{vec}_{\triangle}(R^{t})\right)`,
    "label-alignment": String.raw`\operatorname{LA}=\operatorname{corr}\!\left(\{\operatorname{RDM}_{ij}\}_{i<j},\{\mathbf 1[y_i\neq y_j]\}_{i<j}\right)`,
    "fixed-anchor": String.raw`\mathcal A=\{x_1,\ldots,x_n\}\ \text{fixed},\qquad H_{t_1}(\mathcal A)\ \text{is compared with}\ H_{t_2}(\mathcal A)`,
    "checkpoint": String.raw`\mathcal C_t=(\theta_t,s_t),\qquad m_t=\mathcal M(\mathcal C_t,\mathcal D_{\mathrm{eval}})`,
    "task-boundary": String.raw`\operatorname{Train}(\mathcal D_t)\ \longrightarrow\ \operatorname{Evaluate}(\mathcal D_1,\ldots,\mathcal D_T)\ \longrightarrow\ \operatorname{Train}(\mathcal D_{t+1})`,
    "random-seed": String.raw`\omega_s=\operatorname{PRNG}(s),\qquad r_s=F(\omega_s,\mathcal P)`,
    "paired-design": String.raw`d_s=m_s^{A}-m_s^{B},\qquad \bar d=\frac{1}{S}\sum_{s=1}^{S}d_s`,
    "validation-set": String.raw`\widehat\theta=\underset{\theta\in\mathcal C}{\arg\min}\;\mathcal L_{\mathrm{val}}(\theta),\qquad \text{report }\mathcal L_{\mathrm{test}}(\widehat\theta)`,
    "test-set": String.raw`\widehat{\mathcal R}_{\mathrm{test}}(\widehat\theta)=\frac{1}{|\mathcal D_{\mathrm{test}}|}\sum_{(x,y)\in\mathcal D_{\mathrm{test}}}\ell\!\left(f_{\widehat\theta}(x),y\right)`,
    "epoch": String.raw`N_{\mathrm{updates/epoch}}\approx\left\lceil\frac{N}{B}\right\rceil`,
    "batch-size": String.raw`\mathcal L_{\mathcal B}(\theta)=\frac{1}{|\mathcal B|}\sum_{i\in\mathcal B}\ell_i(\theta)`,
    "parameter-count": String.raw`P_{\mathrm{dense}}=d_{\mathrm{out}}(d_{\mathrm{in}}+1),\qquad P_{\mathrm{conv}}=c_{\mathrm{out}}(c_{\mathrm{in}}k_hk_w+1)`,
    "wall-clock-time": String.raw`t_{\mathrm{elapsed}}=t_{\mathrm{stop}}^{(\mathrm{sync})}-t_{\mathrm{start}}^{(\mathrm{sync})}`,
    "memory-footprint": String.raw`M_{\mathrm{train}}\approx M_{\mathrm{params}}+M_{\mathrm{grads}}+M_{\mathrm{optimizer}}+M_{\mathrm{activations}}+M_{\mathrm{temporary}}`,
    "confounding": String.raw`P(Y\mid X)\neq P\!\left(Y\mid\operatorname{do}(X)\right)\quad\text{when common causes }C\text{ are uncontrolled}`,
    "mediation": String.raw`\operatorname{TE}=\mathbb E[Y\mid\operatorname{do}(X=1)]-\mathbb E[Y\mid\operatorname{do}(X=0)],\qquad X\rightarrow M\rightarrow Y`,
    "causal-claim": String.raw`\operatorname{ACE}=\mathbb E[Y\mid\operatorname{do}(X=x_1)]-\mathbb E[Y\mid\operatorname{do}(X=x_0)]`,
    "observational-result": String.raw`\Delta_{\mathrm{obs}}=\mathbb E[Y\mid X=a]-\mathbb E[Y\mid X=b]`,
    "model-width": String.raw`P_{\mathrm{dense}}(w)=\mathcal O(d_{\mathrm{in}}w+w^2+wd_{\mathrm{out}})`,
    "model-depth": String.raw`f_\theta=f_L\circ f_{L-1}\circ\cdots\circ f_1`
  };

  const esc = value => String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);
  const sourceFor = key => registry.sources[key];
  let activeCategory = "All";
  let query = "";

  function sourceLink(key) {
    const source = sourceFor(key);
    if (!source) return "";
    const external = /^https?:/.test(source.href);
    return `<li><span class="source-kind">${esc(source.kind)}</span> · <a href="${esc(source.href)}"${external ? ' target="_blank" rel="noreferrer"' : ""}>${esc(source.label)}</a></li>`;
  }

  function termEntry(term) {
    const latex = LATEX[term.id] || String.raw`\text{No LaTeX definition registered}`;
    const related = (term.related || []).map(id => {
      const match = registry.terms.find(candidate => candidate.id === id);
      return match ? `<a href="#${esc(id)}">${esc(match.name)}</a>` : "";
    }).join("");
    return `<article class="term-entry" id="${esc(term.id)}" data-category="${esc(term.category)}" data-search="${esc([term.name,...term.aliases,term.category,term.summary,term.definition].join(" ").toLowerCase())}">
      <div class="term-head"><div><span class="term-category">${esc(term.category)}</span><h2>${esc(term.name)}</h2><p class="term-summary">${esc(term.summary)}</p></div><a class="permalink" href="#${esc(term.id)}" aria-label="Permanent link to ${esc(term.name)}"># Permalink</a></div>
      <div class="term-body">
        <section class="definition-panel"><h3>Rigorous definition</h3><p>${esc(term.definition)}</p></section>
        <aside class="project-panel"><h3>How this project uses it</h3><p>${esc(term.project)}</p>${related ? `<nav class="related" aria-label="Related terms">${related}</nav>` : ""}</aside>
        <section class="formula-panel"><h3>Mathematical definition</h3><div class="formula" role="math" aria-label="${esc(term.formula)}">\\[${esc(latex)}\\]</div><p class="formula-note">${esc(term.formulaNote)}</p><details class="latex-source"><summary>Copy LaTeX source</summary><code>${esc(latex)}</code></details></section>
        <section class="sources-panel"><h3>Sources</h3><ol class="sources-list">${term.sources.map(sourceLink).join("")}</ol></section>
      </div>
    </article>`;
  }

  function filteredTerms() {
    return registry.terms.filter(term => {
      const categoryMatch = activeCategory === "All" || term.category === activeCategory;
      const haystack = [term.name,...term.aliases,term.category,term.summary,term.definition].join(" ").toLowerCase();
      return categoryMatch && (!query || haystack.includes(query));
    });
  }

  function renderTerms() {
    const terms = filteredTerms();
    document.getElementById("term-index").innerHTML = terms.map(term => `<a class="term-chip" href="#${esc(term.id)}">${esc(term.name)}</a>`).join("");
    document.getElementById("terms-list").innerHTML = terms.length ? terms.map(termEntry).join("") : '<div class="empty-results">No definitions match this filter.</div>';
    document.getElementById("wiki-count").textContent = `${terms.length} of ${registry.terms.length} source-backed definitions`;
    if (window.MathJax?.typesetPromise) window.MathJax.typesetPromise([document.getElementById("terms-list")]).catch(() => {});
  }

  function setupCategories() {
    const categories = ["All", ...new Set(registry.terms.map(term => term.category))];
    const nav = document.getElementById("category-nav");
    nav.innerHTML = categories.map(category => `<button type="button" data-category="${esc(category)}" aria-pressed="${category === "All"}">${esc(category)}</button>`).join("");
    nav.addEventListener("click", event => {
      const button = event.target.closest("button[data-category]");
      if (!button) return;
      activeCategory = button.dataset.category;
      nav.querySelectorAll("button").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
      renderTerms();
    });
  }

  function setupReturnNavigation() {
    const params = new URLSearchParams(location.search);
    const requested = params.get("return");
    const safeReturn = requested && requested.includes("/.research/") ? requested : "../learning-without-letting-go.html";
    let stored = null;
    try { stored = JSON.parse(sessionStorage.getItem("research-wiki-return") || "null"); } catch (_) {}
    document.querySelectorAll("[data-return-link]").forEach(link => {
      link.href = safeReturn;
      const fullLabel = stored?.title ? `← Back to ${stored.title}` : "← Back to source page";
      link.textContent = matchMedia("(max-width: 560px)").matches ? "← Back to source" : fullLabel;
      link.title = fullLabel;
      link.addEventListener("click", () => {
        if (!stored) return;
        try { sessionStorage.setItem("research-wiki-restore", JSON.stringify(stored)); } catch (_) {}
      });
    });
  }

  document.getElementById("wiki-search").addEventListener("input", event => {
    query = event.target.value.trim().toLowerCase();
    renderTerms();
  });
  document.getElementById("theme-button").addEventListener("click", () => {
    document.documentElement.style.colorScheme = document.documentElement.style.colorScheme === "light" ? "dark" : "light";
  });
  setupCategories();
  setupReturnNavigation();
  renderTerms();
})();
