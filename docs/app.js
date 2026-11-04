// Static explorer for the TBI / Alzheimer's AD-likeness scores. No dependencies.
const REG = { TCx: "Temporal cortex", PCx: "Parietal cortex", HIP: "Hippocampus", FWM: "White matter" };
const NS = "http://www.w3.org/2000/svg";
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const $ = (id) => document.getElementById(id);
const fmt = (x, d = 2) => (x > 0 ? "+" : "") + x.toFixed(d);
let S = [], G = {}, SUM = {}, selected = null;

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function hash(str) { let h = 2166136261; for (const c of str) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return (h >>> 0) / 4294967295; }

// Diverging colour for AD z: cool (down in AD) -> neutral -> accent (up in AD).
function divColor(z, lim = 8) {
  const t = Math.max(-1, Math.min(1, z / lim));
  const mix = (a, b, f) => a.map((x, i) => Math.round(x + (b[i] - x) * f));
  const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const mid = hex(css("--surface").length === 7 ? css("--surface") : "#ffffff");
  const c = t >= 0 ? mix(mid, hex(css("--accent")), t) : mix(mid, hex(css("--cool")), -t);
  return `rgb(${c.join(",")})`;
}

function cards() {
  const t = SUM.tbi.lr_l2, cv = SUM.cv;
  const items = [
    [`${fmt(t.pair)} SD`, "TBI vs matched non-TBI donor, AD-likeness score (logistic regression)", true],
    [`p = ${t.signflip_p}`, "exact matched-pair sign-flip test (48 pairs)", false],
    [`${t.random_pct}%`, "of random gene signatures show a smaller TBI shift", false],
    [`${cv.gat_string.auroc} vs ${cv.lr_l2.auroc}`, "AUROC: GAT on STRING vs logistic regression", false],
  ];
  $("cards").innerHTML = items.map(([n, l, hi]) =>
    `<div class="card${hi ? " hi" : ""}"><div class="num">${n}</div><div class="lbl">${l}</div></div>`).join("");
  const k = {
    adsig: SUM.ad_signature.fdr05_consistent_3of3.toLocaleString("en"), repl: Math.round(SUM.replication_sign_agreement * 100) + "%",
    globp: SUM.gene_level_global_p, pair: fmt(t.pair), sfp: t.signflip_p, beta: fmt(t.beta), mp: t.p,
    rand: t.random_pct, lr: cv.lr_l2.auroc, gat: cv.gat_string.auroc, gatr: cv.gat_random.auroc, gatn: cv.gat_none.auroc,
  };
  document.querySelectorAll("[data-k]").forEach((e) => (e.textContent = k[e.dataset.k]));
}

function rows() {
  const r = $("region").value;
  return r === "all" ? Object.keys(REG) : [r];
}

function drawSwarm() {
  const model = $("model").value, regs = rows();
  const box = $("swarm");
  // Draw at the container's own width so text stays readable on phones.
  const W = Math.max(300, Math.min(760, box.clientWidth || 720));
  const narrow = W < 520;
  const rowH = regs.length === 1 ? 150 : (narrow ? 84 : 92), top = 18, right = 12;
  const left = regs.length === 1 ? (narrow ? 50 : 64) : (narrow ? 84 : 120);
  const H = top + regs.length * rowH + 36;
  const vals = S.map((d) => d.s[model]);
  const lo = Math.floor(Math.min(...vals)), hi = Math.ceil(Math.max(...vals));
  const x = (v) => left + ((v - lo) / (hi - lo)) * (W - left - right);
  box.innerHTML = "";
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
  const ax = el("g", { class: "axis" }, svg);
  for (let v = lo; v <= hi; v++) {
    el("line", { x1: x(v), x2: x(v), y1: top - 6, y2: H - 30, "stroke-dasharray": v === 0 ? "" : "2 4" }, ax);
    el("text", { x: x(v), y: H - 12, "text-anchor": "middle" }, ax).textContent = v;
  }
  el("text", { x: (left + W) / 2, y: H - 1, "text-anchor": "middle", class: "lab" }, svg).textContent = "AD-likeness score (SD) → more Alzheimer-like";
  const colors = { 1: css("--accent"), 0: css("--neutral") };
  const stats = [];
  regs.forEach((reg, i) => {
    const y0 = top + i * rowH;
    // Multi-region view: label the row with the region. Single-region view: label the two groups.
    if (regs.length > 1) {
      el("text", { x: 0, y: y0 + rowH / 2 + 4, class: "lab" }, svg).textContent = narrow ? reg : REG[reg];
    }
    [0, 1].forEach((tbi) => {
      const yc = y0 + (tbi ? rowH * 0.3 : rowH * 0.7);
      if (regs.length === 1) el("text", { x: 0, y: yc + 4, class: "lab" }, svg).textContent = tbi ? "TBI" : "no TBI";
      const group = S.filter((d) => d.region === reg && d.tbi === tbi);
      const band = rowH * 0.18;
      group.forEach((d) => {
        const cy = yc + (hash(d.id) - 0.5) * 2 * band;
        const c = el("circle", { cx: x(d.s[model]), cy, r: regs.length === 1 ? 5 : (narrow ? 3 : 3.6), fill: colors[tbi],
          "fill-opacity": 0.75, class: "dot" + (d.id === selected ? " sel" : ""), tabindex: -1 }, svg);
        c.addEventListener("click", () => select(d.id));
        el("title", {}, c).textContent = `${d.donor} · ${tbi ? "TBI" : "no TBI"} · ${d.dx} · ${fmt(d.s[model])} SD`;
      });
      const m = group.reduce((a, d) => a + d.s[model], 0) / group.length;
      stats.push([reg, tbi, m, group.length]);
      el("line", { x1: x(m), x2: x(m), y1: yc - band - 6, y2: yc + band + 6, stroke: colors[tbi], class: "meanline" }, svg);
    });
    if (i < regs.length - 1) el("line", { x1: 0, x2: W, y1: y0 + rowH, y2: y0 + rowH, stroke: css("--rule") }, svg);
  });
  const per = regs.map((r) => {
    const t = stats.find((s) => s[0] === r && s[1] === 1), n = stats.find((s) => s[0] === r && s[1] === 0);
    return `${REG[r]}: TBI ${fmt(t[2])} (n=${t[3]}) vs no TBI ${fmt(n[2])} (n=${n[3]})`;
  });
  $("swarm-cap").innerHTML = `<span class="sw pos"></span> TBI &nbsp; <span class="sw" style="background:${css("--neutral")}"></span> no TBI. ` +
    `Vertical bars are group means. ${per.join("; ")}.`;
}

function fillSamples() {
  const reg = $("region").value, sel = $("sample");
  const list = S.filter((d) => reg === "all" || d.region === reg)
    .sort((a, b) => a.donor.localeCompare(b.donor) || a.region.localeCompare(b.region));
  sel.innerHTML = list.map((d) => `<option value="${d.id}">${d.donor} · ${d.region} · ${d.tbi ? "TBI" : "no TBI"} · ${d.dx}</option>`).join("");
  if (!list.some((d) => d.id === selected)) selected = list[0].id;
  sel.value = selected;
}

function select(id) {
  selected = id; $("sample").value = id;
  const d = S.find((s) => s.id === id), model = $("model").value;
  const same = S.filter((s) => s.region === d.region), pct = Math.round(100 * same.filter((s) => s.s[model] < d.s[model]).length / same.length);
  $("sample-head").innerHTML = `<div class="big">${fmt(d.s[model])} <span style="font-size:16px">SD</span></div>
    <div class="tiny">${pct}th percentile in ${REG[d.region].toLowerCase()} · ${$("model").selectedOptions[0].text}</div>
    <dl class="meta"><dt>Donor</dt><dd>${d.donor}</dd><dt>Region</dt><dd>${REG[d.region]}</dd>
    <dt>TBI with LOC</dt><dd>${d.tbi ? "yes" : "no"}</dd><dt>Clinical diagnosis</dt><dd>${d.dx === "other_dementia" ? "other dementia" : d.dx}</dd>
    <dt>Braak stage</dt><dd>${d.braak ?? "–"}</dd><dt>Age / sex</dt><dd>${d.age ?? "–"} / ${d.sex}</dd></dl>`;
  const mx = Math.max(...d.g.map((g) => Math.abs(g[1])));
  $("contrib").innerHTML = `<div class="bars">` + d.g.map(([g, v]) => {
    const w = (Math.abs(v) / mx) * 50, col = v > 0 ? css("--accent") : css("--cool");
    const style = v > 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
    return `<span>${g}</span><div class="bartrack"><div class="mid"></div><div class="bar" style="position:absolute;${style};background:${col}" title="${v}"></div></div>`;
  }).join("") + `</div>`;
  drawSwarm();
}

function drawEgo() {
  const g = $("gene").value, d = G[g], box = $("ego"); box.innerHTML = "";
  const W = 760, H = 520, cx = W / 2, cy = H / 2, R = 200;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
  const all = [[g, d.coef, d.adz], ...d.n];
  const mx = Math.max(...all.map((n) => Math.abs(n[1] ?? 0)), 1e-6);
  const size = (c) => 7 + 20 * Math.sqrt(Math.abs(c ?? 0) / mx);
  const pos = d.n.map((_, i) => {
    const a = (2 * Math.PI * i) / Math.max(d.n.length, 1) - Math.PI / 2;
    return [cx + R * Math.cos(a), cy + R * 0.92 * Math.sin(a)];
  });
  pos.forEach(([x, y]) => el("line", { x1: cx, y1: cy, x2: x, y2: y }, svg));
  d.n.forEach(([n, c, z], i) => {
    const [x, y] = pos[i];
    const circ = el("circle", { cx: x, cy: y, r: size(c), fill: divColor(z ?? 0), stroke: css("--rule") }, svg);
    el("title", {}, circ).textContent = `${n}: AD z ${z ?? "n/a"}, weight ${c ?? "n/a"}`;
    const below = y > cy;
    el("text", { x, y: y + (below ? size(c) + 13 : -size(c) - 6), "text-anchor": "middle" }, svg).textContent = n;
  });
  el("circle", { cx, cy, r: size(d.coef) + 4, fill: divColor(d.adz ?? 0), stroke: css("--ink"), "stroke-width": 2 }, svg);
  el("text", { x: cx, y: cy + 4, "text-anchor": "middle", "font-weight": 600, "font-size": 13 }, svg).textContent = g;
  $("ego-cap").textContent = d.deg
    ? `${g}: Alzheimer meta z ${d.adz}, score weight ${d.coef} (${d.coef > 0 ? "pushes toward" : "pushes away from"} Alzheimer's). ` +
      `${d.deg} STRING partners among the 4,000 model genes; showing ${d.n.length} with the largest weight.`
    : `${g} has no high-confidence STRING partner among the 4,000 model genes.`;
}

async function init() {
  [S, G, SUM] = await Promise.all(["data/samples.json", "data/genes.json", "data/summary.json"].map((u) => fetch(u).then((r) => r.json())));
  cards();
  $("gene").innerHTML = Object.keys(G).map((g) => `<option>${g}</option>`).join("");
  $("gene").value = G.GFAP ? "GFAP" : Object.keys(G)[0];
  $("model").addEventListener("change", () => select(selected));
  $("region").addEventListener("change", () => { fillSamples(); select(selected); });
  $("sample").addEventListener("change", (e) => select(e.target.value));
  $("gene").addEventListener("change", drawEgo);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { select(selected); drawEgo(); });
  let rt; let lastW = $("swarm").clientWidth;
  addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => { const w = $("swarm").clientWidth; if (Math.abs(w - lastW) > 20) { lastW = w; drawSwarm(); } }, 150);
  });
  fillSamples(); select(selected); drawEgo();
}
init().catch((e) => { $("swarm").textContent = "Could not load data: " + e.message; });
