function esc(value) {
  return String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
}

function highlightKeyTerms(value) {
  const pattern = /([+−-]?\d+(?:\.\d+)?\s*(?:pp|%|×)|±\s*\d+(?:\.\d+)?\s*(?:pp|%)|\bT\s*=\s*(?:\{[^}]+\}|\d+)|\b(?:PC|RBP|BP)\b|\b(?:not run|not established|not achieved|rejected|frozen|no replay|catastrophic forgetting|under-learning)\b|stability[–-]plasticity trade[–-]off)/gi;
  return esc(value).replace(pattern, '<strong class="key-term">$1</strong>');
}

function renderModelMap(card) {
  if (!card.flow) return "";
  const stages = card.flow.map((item, index) => `<div class="architecture-stage ${index === 0 || index === card.flow.length - 1 ? "terminal" : "operation"} ${index === card.flow.length - 1 ? "last" : ""}" style="grid-row:${index + 1}"><span>${String(index + 1).padStart(2, "0")}</span><strong>${esc(item)}</strong></div>`).join("");
  const latent = (card.latent || []).map((item, index) => `<div class="error-coupling" style="grid-row:${index + 2}" aria-hidden="true"><span>ε${index + 1}</span></div><div class="latent-node" style="grid-row:${index + 2}">${esc(item)}</div>`).join("");
  return `<div class="model-map" aria-label="${esc(card.title)} architecture">
    <div class="paper-diagram-head"><span>Learned forward path</span><span>error</span><span>PC inferred states</span></div>
    <div class="paper-network">${stages}${latent}</div>
    <div class="model-legend"><span><i class="forward-swatch"></i>learned forward path</span><span><i class="latent-swatch"></i>inferred state, not an extra feedforward layer</span></div>
  </div>`;
}

function renderSettingsCard(card, type) {
  const rows = card.settings.map(([label, value]) => `<div class="setting-row"><dt>${esc(label)}</dt><dd>${highlightKeyTerms(value)}</dd></div>`).join("");
  const diagram = type === "model" ? renderModelMap(card) : "";
  return `<article class="settings-card ${type}"><span class="card-kicker">${type === "model" ? "Model" : "Experiment"}</span><h3>${esc(card.title)}</h3><p>${esc(card.note)}</p>${diagram}<dl class="setting-list">${rows}</dl></article>`;
}

function renderResultCard(result) {
  const label = result.kind === "achieved" ? "Achieved" : result.kind === "non-achieved" ? "Non-achieved" : "Boundary";
  return `<article class="result-card ${esc(result.kind)}"><span class="claim-label ${result.kind === "achieved" ? "verified" : result.kind === "non-achieved" ? "rejected" : "observation"}">${label}</span><h3>${esc(result.title)}</h3><span class="metric">${esc(result.metric)}</span><p>${highlightKeyTerms(result.text)}</p></article>`;
}

function renderTable(table) {
  if (!table) return "";
  return `<div class="result-table-wrap"><table class="result-table"><thead><tr>${table.headers.map(item => `<th>${esc(item)}</th>`).join("")}</tr></thead><tbody>${table.rows.map(row => `<tr>${row.map(item => `<td>${esc(item)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function svgFrame(chart) {
  const width = 920, height = 380, left = 70, right = 24, top = 24, bottom = 62;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const yMin = chart.yMin, yMax = chart.yMax;
  const y = value => top + plotHeight - ((value - yMin) / (yMax - yMin)) * plotHeight;
  const x = index => left + (chart.x.length === 1 ? plotWidth / 2 : index / (chart.x.length - 1) * plotWidth);
  const ticks = Array.from({length: 5}, (_, index) => yMin + (yMax - yMin) * index / 4);
  let axes = `<line x1="${left}" y1="${top + plotHeight}" x2="${left + plotWidth}" y2="${top + plotHeight}" stroke="var(--muted)"/><line x1="${left}" y1="${top}" x2="${left}" y2="${top + plotHeight}" stroke="var(--muted)"/>`;
  ticks.forEach(tick => {
    axes += `<line x1="${left}" y1="${y(tick)}" x2="${left + plotWidth}" y2="${y(tick)}" stroke="var(--grid)"/><text x="${left - 9}" y="${y(tick) + 4}" text-anchor="end" fill="var(--muted)" font-size="11">${Math.abs(tick) < 1 && tick !== 0 ? tick.toFixed(3) : tick.toFixed(1)}</text>`;
  });
  const labelIndexes = [...new Set([0, Math.floor((chart.x.length - 1) / 4), Math.floor((chart.x.length - 1) / 2), Math.floor((chart.x.length - 1) * .75), chart.x.length - 1])];
  labelIndexes.forEach(index => { axes += `<text x="${x(index)}" y="${top + plotHeight + 22}" text-anchor="middle" fill="var(--muted)" font-size="11">${esc(chart.x[index])}</text>`; });
  axes += `<text x="${left + plotWidth / 2}" y="${height - 8}" text-anchor="middle" fill="var(--muted)" font-size="11">${esc(chart.xLabel)}</text><text transform="translate(15 ${top + plotHeight / 2}) rotate(-90)" text-anchor="middle" fill="var(--muted)" font-size="11">${esc(chart.yLabel)}</text>`;
  return {width, height, left, top, plotWidth, plotHeight, x, y, axes};
}

function renderLineChart(chart) {
  const frame = svgFrame(chart);
  let marks = "";
  chart.series.forEach(series => {
    const points = series.y.map((value, index) => `${frame.x(index)},${frame.y(value)}`).join(" ");
    marks += `<polyline points="${points}" fill="none" stroke="${series.color}" stroke-width="2.5" ${series.dash ? 'stroke-dasharray="7 5"' : ""}/>`;
    marks += series.y.map((value, index) => `<circle cx="${frame.x(index)}" cy="${frame.y(value)}" r="3" fill="${series.color}"><title>${esc(series.name)} · ${esc(chart.x[index])}: ${value}</title></circle>`).join("");
  });
  return `<svg viewBox="0 0 ${frame.width} ${frame.height}" role="img" aria-label="${esc(chart.title)}">${frame.axes}${marks}</svg>`;
}

function renderBarChart(chart) {
  const frame = svgFrame(chart);
  const groupWidth = frame.plotWidth / chart.x.length;
  const barWidth = Math.min(46, groupWidth * .75 / chart.series.length);
  const baseline = frame.y(Math.max(0, chart.yMin));
  let marks = "";
  chart.x.forEach((label, groupIndex) => {
    const center = frame.left + groupWidth * (groupIndex + .5);
    chart.series.forEach((series, seriesIndex) => {
      const value = series.y[groupIndex];
      const valueY = frame.y(value);
      const x = center + (seriesIndex - (chart.series.length - 1) / 2) * (barWidth + 3) - barWidth / 2;
      const top = Math.min(valueY, baseline), height = Math.max(1, Math.abs(baseline - valueY));
      marks += `<rect x="${x}" y="${top}" width="${barWidth}" height="${height}" rx="4" fill="${series.color}"><title>${esc(series.name)} · ${esc(label)}: ${value}</title></rect>`;
    });
  });
  return `<svg viewBox="0 0 ${frame.width} ${frame.height}" role="img" aria-label="${esc(chart.title)}">${frame.axes}${marks}</svg>`;
}

function renderChart(chart, index, charts) {
  const svg = chart.type === "bar" ? renderBarChart(chart) : renderLineChart(chart);
  const legend = chart.series.map(series => `<span><i style="background:${series.color}"></i>${esc(series.name)}${series.dash ? " · dashed" : ""}</span>`).join("");
  return `<article class="chart-card ${index === 0 && charts.length > 1 ? "featured" : ""}"><div class="chart-title"><span class="chart-index">Figure ${String(index + 1).padStart(2, "0")}</span><div><h3>${esc(chart.title)}</h3><p>${highlightKeyTerms(chart.subtitle)}</p></div></div><div class="chart">${svg}</div><div class="chart-legend">${legend}</div>${chart.takeaway ? `<div class="chart-takeaway"><strong>Read this as</strong>${highlightKeyTerms(chart.takeaway)}</div>` : ""}</article>`;
}

function renderClaim(claim) {
  const labels = {hypothesis:"Hypothesis", verified:"Verified", observation:"Observation", interpretation:"Interpretation", rejected:"Rejected", open:"Open question"};
  return `<article class="claim-card"><span class="claim-label ${esc(claim.type)}">${esc(labels[claim.type] || claim.type)}</span><div><h3>${esc(claim.title)}</h3><p>${highlightKeyTerms(claim.text)}</p></div></article>`;
}

function graphNodeForPage(id) {
  return {
    "cifar-figure-4i":"exp-cifar-repro", "static-mnist-relaxation":"exp-static-relaxation",
    "permuted-mnist-relaxation":"exp-permuted-relaxation", "split-mnist-relaxation":"exp-split-relaxation",
    "width-intervention":"exp-width-scale", "depth-intervention":"exp-depth-scale",
    "split-cifar-behavior":"exp-split-cifar", "split-cifar-representations":"exp-representations",
    "fair-budget-comparator":"next-fair-budget", "update-alignment-snr":"next-update-diagnostics",
    "acquisition-matched-causality":"next-plasticity-match", "depth-relaxation":"next-depth-relaxation"
  }[id];
}

function firstClaim(item, types) {
  return item.claims.find(claim => types.includes(claim.type));
}

function renderStorySpine(item) {
  const hypothesis = firstClaim(item, ["hypothesis"]);
  const observation = firstClaim(item, ["verified", "observation"]);
  const verdict = firstClaim(item, ["rejected"]);
  const uncertainty = firstClaim(item, ["open", "interpretation"]);
  const steps = item.proposed ? [
    ["Question", hypothesis?.text || item.summary, "question"],
    ["Planned test", `${item.experiment.title}. ${item.experiment.note}`, "test"],
    ["Falsifier", firstClaim(item, ["open"])?.text || "The falsifying outcome must be registered before execution.", "evidence"],
    ["Evidence boundary", observation?.text || "Not run; no outcome is included in backed claims.", "boundary"],
    ["Next", "Execute the registered design before adding any result or learning curve.", "next"]
  ] : [
    ["Question", hypothesis?.text || item.summary, "question"],
    ["Test", `${item.experiment.title}. ${item.experiment.note}`, "test"],
    ["Evidence", observation?.text || item.results[0]?.text || item.answer, "evidence"],
    ["Verdict", verdict?.text || item.answer, "verdict"],
    [uncertainty?.type === "open" ? "Next uncertainty" : "Interpretation", uncertainty?.text || "The stated claim boundary remains in force.", "next"]
  ];
  return `<section class="story-spine" aria-labelledby="story-spine-title"><div class="story-spine-head"><span>Experiment story</span><h2 id="story-spine-title">From question to consequence</h2><p>${item.proposed ? "A planned test is separated from evidence that does not yet exist." : "The full page follows the logic of the experiment, not just its output artifacts."}</p></div><div class="story-steps">${steps.map(([label, text, tone], index) => `<article class="story-step ${tone}"><span class="story-number">${String(index + 1).padStart(2, "0")}</span><strong>${esc(label)}</strong><p>${highlightKeyTerms(text)}</p></article>`).join("")}</div></section>`;
}

function renderExperimentPage(id) {
  const item = EXPERIMENTS[id];
  const root = document.getElementById("experiment-page");
  if (!item) {
    root.innerHTML = `<div class="empty-state"><strong>Experiment not found</strong><p>The requested experiment ID is not registered.</p></div>`;
    return;
  }

  document.title = `${item.short} · Learning Without Letting Go`;
  document.body.dataset.status = item.statusClass;
  document.body.dataset.proposed = String(item.proposed);
  const summaryUrl = `../learning-without-letting-go.html#node=${encodeURIComponent(graphNodeForPage(id))}&view=journey`;
  document.querySelectorAll("[data-summary-link]").forEach(link => link.href = summaryUrl);

  const index = EXPERIMENT_ORDER.indexOf(id);
  const previousId = EXPERIMENT_ORDER[index - 1], nextId = EXPERIMENT_ORDER[index + 1];
  const pager = `<nav class="experiment-pager" aria-label="Experiment pages">${previousId ? `<a class="pager-link" href="${previousId}.html"><span>Previous experiment</span><strong>← ${esc(EXPERIMENTS[previousId].short)}</strong></a>` : `<a class="pager-link" href="${summaryUrl}"><span>Return</span><strong>← Research Atlas</strong></a>`}${nextId ? `<a class="pager-link next" href="${nextId}.html"><span>Next experiment</span><strong>${esc(EXPERIMENTS[nextId].short)} →</strong></a>` : `<a class="pager-link next" href="${summaryUrl}"><span>Return</span><strong>Research Atlas →</strong></a>`}</nav>`;

  root.innerHTML = `
    <section class="experiment-hero">
      <div>
        <p class="eyebrow">${esc(item.source)}</p>
        <div class="status-line"><span class="status-pill ${esc(item.statusClass)}">${esc(item.status)}</span><span class="status-pill ${item.proposed ? "info" : "warn"}">${item.proposed ? "Planned experiment" : "Completed experiment"}</span></div>
        <h1>${esc(item.title)}</h1>
        <p class="hero-summary">${highlightKeyTerms(item.summary)}</p>
        <div class="experiment-meta"><span><small>Model</small><strong>${esc(item.model.title)}</strong></span><span><small>Evidence</small><strong>${item.proposed ? "Not run" : `${item.results.length} result statements`}</strong></span><span><small>Visuals</small><strong>${item.charts.length ? `${item.charts.length} evidence figure${item.charts.length === 1 ? "" : "s"}` : "Intentionally absent"}</strong></span></div>
      </div>
      <aside class="hero-answer"><span>${item.proposed ? "Planned answer format" : "Current answer"}</span><div class="answer-text">${highlightKeyTerms(item.answer)}</div><small>${item.proposed ? "No outcome is implied." : "Read within the evidence and claim boundaries below."}</small></aside>
    </section>

    ${renderStorySpine(item)}

    <nav class="reading-nav" aria-label="Experiment page sections">
      <a href="#settings">1 · Setup</a><a href="#results">2 · Outcome</a><a href="#curves">3 · Dynamics</a><a href="#explanation">4 · Meaning</a>
    </nav>

    <section class="content-section" id="settings">
      <div class="section-head"><div><span class="section-index">01 · Experimental setup</span><h2>What was held fixed—and what changed?</h2></div><p>The model, intervention, stream, controls and evaluation budget sit together so the causal comparison and its limits are visible before the result.</p></div>
      <div class="settings-grid">${renderSettingsCard(item.model, "model")}${renderSettingsCard(item.experiment, "experiment")}</div>
    </section>

    <section class="content-section" id="results">
      <div class="section-head"><div><span class="section-index">02 · Observed outcome</span><h2>${item.proposed ? "What evidence is still missing?" : "What happened—and what did not?"}</h2></div><p>${item.proposed ? "The empty result is intentional: a planned design is not presented as completed evidence." : "Positive findings, negative results and claim boundaries have equal visual weight. A lower forgetting score is never read alone."}</p></div>
      <div class="results-grid">${item.results.map(renderResultCard).join("")}${renderTable(item.table)}</div>
    </section>

    <section class="content-section" id="curves">
      <div class="section-head"><div><span class="section-index">03 · Evidence dynamics</span><h2>${item.proposed ? "No trajectory before execution" : "How did the result unfold?"}</h2></div><p>Learning trajectories, intervention comparisons and representation diagnostics remain attached to the experiment that produced them.</p></div>
      <div class="charts-grid">${item.charts.length ? item.charts.map(renderChart).join("") : `<div class="empty-state"><strong>Not run · no curves yet</strong><p>This proposed experiment has no observed trajectory or result visualization. Curves will be added only after a registered run produces evidence.</p></div>`}</div>
    </section>

    <section class="content-section" id="explanation">
      <div class="section-head"><div><span class="section-index">04 · Scientific reading</span><h2>What can we conclude?</h2></div><p>Hypotheses, verified observations, interpretations, rejected expectations and open mechanisms are labelled separately so the claim ceiling stays visible.</p></div>
      <div class="claims-list">${item.claims.map(renderClaim).join("")}</div>
    </section>

    <section class="evidence-section"><article class="evidence-card"><h2>Evidence and reproduction sources</h2><div class="evidence-links"><a href="../wiki/index.html">Research Wiki →</a>${item.sources.map(([label, href]) => `<a href="${esc(href)}">${esc(label)} →</a>`).join("")}</div></article>${pager}</section>`;
}

document.getElementById("theme-button").addEventListener("click", () => {
  document.documentElement.style.colorScheme = document.documentElement.style.colorScheme === "light" ? "dark" : "light";
});

renderExperimentPage(document.body.dataset.experiment);
