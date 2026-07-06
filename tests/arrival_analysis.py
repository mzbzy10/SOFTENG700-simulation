"""
Rerunnable diagnostic for the per-slice arrival process (eMBB / URLLC / mMTC).

Drives Simulator.generate_embb_arrivals / generate_urllc_arrivals / generate_mmtc_arrivals
directly and discards each step's tasks immediately after recording them - nothing is ever
served, so this is a pure look at arrival rates, task sizes, deadlines, and eMBB's ON/OFF
bursting. Discarding tasks each step (instead of running them through Simulator.run() with
a zero-PRB allocator) keeps this O(steps) - a real unserved backlog would grow every step
forever, making every per-step scan O(steps^2).

Tweak the parameters below and rerun:

    python tests/arrival_analysis.py

Writes tests/arrival_dashboard.html and opens it in your browser.
"""
import json
import os
import sys
import webbrowser

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Simulator import Simulator
from Allocators.FixedAllocator import FixedAllocator

# ---- parameters you can change ----
STEPS = 10000
ZOOM_STEPS = 300       # window shown in the raw "first N steps" timeline
ROLL_WINDOW = 100      # smoothing window for the long-horizon rate chart
ARRIVAL_RATE = 5       # passed to Simulator (URLLC/mMTC placeholder rate)
SEED = 42
OPEN_IN_BROWSER = True
# ------------------------------------

OUTPUT_HTML = os.path.join(os.path.dirname(__file__), "arrival_dashboard.html")


def rolling_rate(arr, window):
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    stride = max(1, window // 4)
    return smoothed[::stride]


def run():
    np.random.seed(SEED)

    sim = Simulator(FixedAllocator(total_prb=0), steps=STEPS, arrival_rate=ARRIVAL_RATE)
    slices = sim.slice_names

    counts = {s: np.zeros(STEPS, dtype=int) for s in slices}
    size_bins = {s: {} for s in slices}
    embb_on_hist = np.zeros(STEPS, dtype=bool)

    for t in range(STEPS):
        sim.time = t
        sim.generate()
        embb_on_hist[t] = sim.embb_on

        for task in sim.requests:
            counts[task.slice_type][t] += 1
            size_bins[task.slice_type][task.size] = size_bins[task.slice_type].get(task.size, 0) + 1

        sim.requests = []  # nothing served -> discard immediately, avoid O(steps^2) growth

    rate_series = {s: rolling_rate(counts[s], ROLL_WINDOW).round(3).tolist() for s in slices}
    rate_x = (np.arange(len(rate_series[slices[0]])) * max(1, ROLL_WINDOW // 4) + ROLL_WINDOW).tolist()

    zoom_n = min(ZOOM_STEPS, STEPS)
    zoom_counts = {s: counts[s][:zoom_n].tolist() for s in slices}
    zoom_embb_on = embb_on_hist[:zoom_n].astype(int).tolist()

    summary = {}
    for s in slices:
        cfg = sim.slice_config[s]
        total = int(counts[s].sum())
        summary[s] = {
            "total_arrivals": total,
            "avg_rate": round(total / STEPS, 3),
            "size_min": cfg["size_range"][0],
            "size_max": cfg["size_range"][1],
            "deadline": cfg["deadline"],
        }

    payload = {
        "steps": STEPS,
        "zoom_steps": zoom_n,
        "roll_window": ROLL_WINDOW,
        "slices": slices,
        "rate_x": rate_x,
        "rate_series": rate_series,
        "zoom_counts": zoom_counts,
        "zoom_embb_on": zoom_embb_on,
        "size_bins": size_bins,
        "summary": summary,
        "embb_on_prob": sim.embb_on_prob,
        "embb_off_prob": sim.embb_off_prob,
        "embb_on_rate": sim.embb_on_rate,
        "embb_off_rate": sim.embb_off_rate,
        "urllc_period": sim.urllc_period,
        "urllc_batch_size": sim.urllc_batch_size,
    }

    for s in slices:
        print(f"{s}: {summary[s]}")

    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload))
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nwrote {OUTPUT_HTML}")

    if OPEN_IN_BROWSER:
        webbrowser.open("file://" + os.path.abspath(OUTPUT_HTML))


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Slice Arrival Process Diagnostic</title>
<style>
  .viz-root {
    --surface-1:      #fcfcfb;
    --page:           #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --gridline:       #e1e0d9;
    --baseline:       #c3c2b7;
    --series-embb:    #2a78d6;
    --series-urllc:   #1baf7a;
    --series-mmtc:    #eda100;
    --wash-embb:      rgba(42,120,214,0.10);
    --tooltip-bg:     #0b0b0b;
    --tooltip-fg:     #fcfcfb;
  }
  @media (prefers-color-scheme: dark) {
    .viz-root {
      --surface-1:      #1a1a19;
      --page:           #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --gridline:       #2c2c2a;
      --baseline:       #383835;
      --series-embb:    #3987e5;
      --series-urllc:   #199e70;
      --series-mmtc:    #c98500;
      --wash-embb:      rgba(57,135,229,0.14);
      --tooltip-bg:     #fcfcfb;
      --tooltip-fg:     #0b0b0b;
    }
  }
  body { margin: 0; background: #f9f9f7; }
  @media (prefers-color-scheme: dark) { body { background: #0d0d0d; } }

  .viz-root {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--page);
    color: var(--text-primary);
    padding: 24px 20px 48px;
    max-width: 1080px;
    margin: 0 auto;
  }
  h1 { font-size: 1.4rem; margin: 0 0 4px; }
  .subtitle { color: var(--text-secondary); font-size: 0.92rem; margin: 0 0 28px; max-width: 68ch; }
  section { margin-bottom: 36px; }
  h2 { font-size: 1.02rem; margin: 0 0 4px; }
  .section-note { color: var(--text-secondary); font-size: 0.85rem; margin: 0 0 14px; max-width: 72ch; }
  .card {
    background: var(--surface-1);
    border: 1px solid var(--gridline);
    border-radius: 10px;
    padding: 16px 18px;
  }

  /* stat tiles */
  .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  .stat-col { background: var(--surface-1); border: 1px solid var(--gridline); border-radius: 10px; padding: 14px 16px; }
  .slice-name { display: flex; align-items: center; gap: 8px; font-weight: 600; margin-bottom: 10px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
  .stat-row { display: flex; justify-content: space-between; padding: 5px 0; border-top: 1px solid var(--gridline); font-size: 0.86rem; }
  .stat-row:first-of-type { border-top: none; }
  .stat-label { color: var(--text-secondary); }
  .stat-value { font-weight: 600; font-variant-numeric: tabular-nums; }

  /* legend */
  .legend { display: flex; gap: 18px; margin-bottom: 10px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.83rem; color: var(--text-secondary); }
  .line-swatch { width: 14px; height: 2px; border-radius: 1px; flex: none; }

  /* zoom panels */
  .zoom-panel { margin-bottom: 4px; }
  .zoom-panel-label { font-size: 0.78rem; color: var(--text-secondary); margin: 0 0 2px 2px; display:flex; justify-content: space-between; }
  svg.chart { width: 100%; height: auto; display: block; overflow: visible; }
  .axis-text { font-size: 9px; fill: var(--text-muted); }
  .grid-line { stroke: var(--gridline); stroke-width: 1; }
  .baseline { stroke: var(--baseline); stroke-width: 1; }

  /* histogram grid */
  .hist-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
  .hist-col { background: var(--surface-1); border: 1px solid var(--gridline); border-radius: 10px; padding: 12px 14px; }
  .hist-title { font-size: 0.85rem; font-weight: 600; margin-bottom: 6px; }

  .tooltip {
    position: absolute;
    pointer-events: none;
    background: var(--tooltip-bg);
    color: var(--tooltip-fg);
    font-size: 0.76rem;
    padding: 6px 9px;
    border-radius: 6px;
    line-height: 1.5;
    white-space: nowrap;
    opacity: 0;
    transform: translate(-50%, -110%);
    transition: opacity 0.08s ease;
    z-index: 10;
  }
  .chart-wrap { position: relative; }

  details.data-table summary { cursor: pointer; font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 10px; }
  table.summary-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  table.summary-table th, table.summary-table td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--gridline); font-variant-numeric: tabular-nums; }
  table.summary-table th { color: var(--text-secondary); font-weight: 600; font-variant-numeric: normal; }

  footer.note { color: var(--text-muted); font-size: 0.78rem; margin-top: 8px; max-width: 72ch; }
</style>
</head>
<body>
<div class="viz-root">
  <h1>Slice arrival process diagnostic</h1>
  <p class="subtitle">
    Null allocator (0 PRBs, nothing served) so every chart below reflects the raw arrival
    process only - task counts, sizes, and deadlines exactly as
    <code>Simulator.generate()</code> produces them, unaffected by serving or backlog dynamics.
  </p>

  <section>
    <h2>Per-slice configuration</h2>
    <p class="section-note">Static parameters and measured long-run averages over the full run.</p>
    <div class="stat-grid" id="stat-grid"></div>
  </section>

  <section>
    <h2 id="zoom-heading">Arrival timeline</h2>
    <p class="section-note">
      Raw per-step arrival counts. The eMBB panel's background wash marks ON bursts
      vs OFF quiet periods; URLLC's alternating pattern is its fixed period; mMTC is
      steady Poisson noise around its mean.
    </p>
    <div class="card">
      <div id="zoom-panels"></div>
    </div>
  </section>

  <section>
    <h2 id="rate-heading">Long-horizon arrival rate</h2>
    <p class="section-note" id="rate-note">Rolling mean arrival rate - hover to inspect a step.</p>
    <div class="card">
      <div class="legend" id="rate-legend"></div>
      <div class="chart-wrap" id="rate-chart-wrap"></div>
    </div>
  </section>

  <section>
    <h2>Task size distribution</h2>
    <p class="section-note">Size of every task generated over the full run, by slice.</p>
    <div class="hist-grid" id="hist-grid"></div>
  </section>

  <section>
    <details class="data-table">
      <summary>Table view - per-slice summary</summary>
      <table class="summary-table" id="summary-table"></table>
    </details>
  </section>

  <footer class="note">
    Generated by tests/arrival_analysis.py - edit the parameters at the top of that file
    and rerun to regenerate this page.
  </footer>
</div>

<script>
const DATA = __DATA_JSON__;

document.getElementById("zoom-heading").textContent = "Arrival timeline - first " + DATA.zoom_steps + " steps";
document.getElementById("rate-heading").textContent = "Long-horizon arrival rate - all " + DATA.steps.toLocaleString() + " steps";
document.getElementById("rate-note").textContent = "Rolling mean arrival rate (window = " + DATA.roll_window + " steps) - hover to inspect a step.";

const COLORS = {
  eMBB:  "var(--series-embb)",
  URLLC: "var(--series-urllc)",
  mMTC:  "var(--series-mmtc)",
};

function scaleLinear(d0, d1, r0, r1) {
  return (v) => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
}

function toSvgPoint(svg, evt) {
  const pt = svg.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  const ctm = svg.getScreenCTM().inverse();
  return pt.matrixTransform(ctm);
}

// ---------- stat tiles ----------
function fmtInt(n) { return n.toLocaleString(); }

const statGrid = document.getElementById("stat-grid");
DATA.slices.forEach((s) => {
  const sum = DATA.summary[s];
  const col = document.createElement("div");
  col.className = "stat-col";
  col.innerHTML = `
    <div class="slice-name"><span class="dot" style="background:${COLORS[s]}"></span>${s}</div>
    <div class="stat-row"><span class="stat-label">Avg arrivals / step</span><span class="stat-value">${sum.avg_rate}</span></div>
    <div class="stat-row"><span class="stat-label">Task size range</span><span class="stat-value">${sum.size_min}–${sum.size_max}</span></div>
    <div class="stat-row"><span class="stat-label">Deadline (steps)</span><span class="stat-value">${sum.deadline}</span></div>
    <div class="stat-row"><span class="stat-label">Total arrivals</span><span class="stat-value">${fmtInt(sum.total_arrivals)}</span></div>
  `;
  statGrid.appendChild(col);
});
if (DATA.summary.eMBB) {
  const embbCol = statGrid.children[0];
  const extra = document.createElement("div");
  extra.className = "stat-row";
  extra.innerHTML = `<span class="stat-label">ON/OFF rate</span><span class="stat-value">${DATA.embb_on_rate}/step · ${DATA.embb_off_rate}/step</span>`;
  embbCol.appendChild(extra);
}
if (DATA.summary.URLLC) {
  const urllcCol = statGrid.children[1];
  const extra = document.createElement("div");
  extra.className = "stat-row";
  extra.innerHTML = `<span class="stat-label">Period</span><span class="stat-value">${DATA.urllc_batch_size} task / ${DATA.urllc_period} steps</span>`;
  urllcCol.appendChild(extra);
}

// ---------- summary table ----------
const table = document.getElementById("summary-table");
table.innerHTML = `
  <thead><tr><th>Slice</th><th>Avg rate</th><th>Size range</th><th>Deadline</th><th>Total arrivals</th></tr></thead>
  <tbody>
    ${DATA.slices.map(s => {
      const sum = DATA.summary[s];
      return `<tr><td>${s}</td><td>${sum.avg_rate}</td><td>${sum.size_min}–${sum.size_max}</td><td>${sum.deadline}</td><td>${fmtInt(sum.total_arrivals)}</td></tr>`;
    }).join("")}
  </tbody>
`;

// ---------- zoom panels ----------
const zoomPanelsEl = document.getElementById("zoom-panels");
const ZW = 900, ZH = 92, ZPAD_L = 28, ZPAD_R = 6, ZPAD_T = 8, ZPAD_B = 16;
const zoomN = DATA.zoom_steps;
const xScaleZoom = scaleLinear(0, zoomN - 1, ZPAD_L, ZW - ZPAD_R);

function runsOf(arr, value) {
  const runs = [];
  let start = null;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === value && start === null) start = i;
    if (arr[i] !== value && start !== null) { runs.push([start, i - 1]); start = null; }
  }
  if (start !== null) runs.push([start, arr.length - 1]);
  return runs;
}

DATA.slices.forEach((s) => {
  const counts = DATA.zoom_counts[s];
  const maxY = Math.max(1, ...counts) * 1.15;
  const yScale = scaleLinear(0, maxY, ZH - ZPAD_B, ZPAD_T);
  const color = COLORS[s];

  let bg = "";
  if (s === "eMBB") {
    const onRuns = runsOf(DATA.zoom_embb_on, 1);
    bg = onRuns.map(([a, b]) => {
      const x0 = xScaleZoom(a - 0.5), x1 = xScaleZoom(b + 0.5);
      return `<rect x="${x0}" y="${ZPAD_T}" width="${x1 - x0}" height="${ZH - ZPAD_T - ZPAD_B}" fill="var(--wash-embb)"/>`;
    }).join("");
  }

  const points = counts.map((c, i) => `${xScaleZoom(i)},${yScale(c)}`).join(" ");
  const areaPoints = `${xScaleZoom(0)},${yScale(0)} ${points} ${xScaleZoom(counts.length - 1)},${yScale(0)}`;

  const gridY = [0, Math.round(maxY / 2), Math.round(maxY)];
  const gridLines = gridY.map(v => {
    const y = yScale(v);
    return `<line class="grid-line" x1="${ZPAD_L}" x2="${ZW - ZPAD_R}" y1="${y}" y2="${y}"/>
            <text class="axis-text" x="4" y="${y + 3}">${v}</text>`;
  }).join("");

  const wrap = document.createElement("div");
  wrap.className = "zoom-panel chart-wrap";
  wrap.innerHTML = `
    <div class="zoom-panel-label"><span><strong style="color:${color}">●</strong> ${s}</span><span>steps 0–${zoomN - 1}</span></div>
    <svg class="chart" viewBox="0 0 ${ZW} ${ZH}" data-slice="${s}">
      ${gridLines}
      ${bg}
      <polygon points="${areaPoints}" fill="${color}" opacity="0.12"/>
      <polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>
      <line class="baseline" x1="${ZPAD_L}" x2="${ZW - ZPAD_R}" y1="${ZH - ZPAD_B}" y2="${ZH - ZPAD_B}"/>
      <line class="hover-line" x1="0" y1="${ZPAD_T}" x2="0" y2="${ZH - ZPAD_B}" stroke="var(--text-muted)" stroke-width="1" opacity="0"/>
    </svg>
  `;
  zoomPanelsEl.appendChild(wrap);

  const svg = wrap.querySelector("svg");
  const hoverLine = wrap.querySelector(".hover-line");
  let tooltip = zoomPanelsEl.parentElement.querySelector(".zoom-tooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.className = "tooltip zoom-tooltip";
    zoomPanelsEl.parentElement.style.position = "relative";
    zoomPanelsEl.parentElement.appendChild(tooltip);
  }

  svg.addEventListener("mousemove", (evt) => {
    const p = toSvgPoint(svg, evt);
    let idx = Math.round((p.x - ZPAD_L) / (ZW - ZPAD_L - ZPAD_R) * (zoomN - 1));
    idx = Math.max(0, Math.min(zoomN - 1, idx));
    hoverLine.setAttribute("x1", xScaleZoom(idx));
    hoverLine.setAttribute("x2", xScaleZoom(idx));
    hoverLine.setAttribute("opacity", "1");
    const rect = svg.getBoundingClientRect();
    const px = rect.left + (xScaleZoom(idx) / ZW) * rect.width;
    const py = rect.top;
    const parentRect = zoomPanelsEl.parentElement.getBoundingClientRect();
    tooltip.style.left = (px - parentRect.left) + "px";
    tooltip.style.top = (py - parentRect.top) + "px";
    tooltip.style.opacity = "1";
    const onState = DATA.zoom_embb_on[idx] ? "ON" : "OFF";
    tooltip.innerHTML = `step ${idx} &middot; ${s} = ${DATA.zoom_counts[s][idx]}` + (s === "eMBB" ? ` &middot; ${onState}` : "");
  });
  svg.addEventListener("mouseleave", () => {
    hoverLine.setAttribute("opacity", "0");
    tooltip.style.opacity = "0";
  });
});

// ---------- long horizon rate chart ----------
const legendEl = document.getElementById("rate-legend");
DATA.slices.forEach((s) => {
  const item = document.createElement("div");
  item.className = "legend-item";
  item.innerHTML = `<span class="line-swatch" style="background:${COLORS[s]}"></span>${s}`;
  legendEl.appendChild(item);
});

const RW = 1000, RH = 260, RPAD_L = 36, RPAD_R = 12, RPAD_T = 10, RPAD_B = 24;
const rateX = DATA.rate_x;
const allRates = DATA.slices.flatMap(s => DATA.rate_series[s]);
const rateMaxY = Math.max(...allRates) * 1.1;
const xScaleRate = scaleLinear(rateX[0], rateX[rateX.length - 1], RPAD_L, RW - RPAD_R);
const yScaleRate = scaleLinear(0, rateMaxY, RH - RPAD_B, RPAD_T);

const rateGridVals = [0, rateMaxY / 2, rateMaxY].map(v => Math.round(v * 10) / 10);
const rateGridLines = rateGridVals.map(v => {
  const y = yScaleRate(v);
  return `<line class="grid-line" x1="${RPAD_L}" x2="${RW - RPAD_R}" y1="${y}" y2="${y}"/>
          <text class="axis-text" x="4" y="${y + 3}">${v}</text>`;
}).join("");
const xTicks = [rateX[0], Math.round(rateX[rateX.length-1]/2), rateX[rateX.length-1]];
const xTickLabels = xTicks.map(v => `<text class="axis-text" x="${xScaleRate(v)}" y="${RH - 6}" text-anchor="middle">${fmtInt(v)}</text>`).join("");

const rateLines = DATA.slices.map((s) => {
  const pts = DATA.rate_series[s].map((v, i) => `${xScaleRate(rateX[i])},${yScaleRate(v)}`).join(" ");
  return `<polyline points="${pts}" fill="none" stroke="${COLORS[s]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
}).join("");

const rateWrap = document.getElementById("rate-chart-wrap");
rateWrap.innerHTML = `
  <svg class="chart" id="rate-svg" viewBox="0 0 ${RW} ${RH}">
    ${rateGridLines}
    ${rateLines}
    <line class="baseline" x1="${RPAD_L}" x2="${RW - RPAD_R}" y1="${RH - RPAD_B}" y2="${RH - RPAD_B}"/>
    ${xTickLabels}
    <line class="hover-line" x1="0" y1="${RPAD_T}" x2="0" y2="${RH - RPAD_B}" stroke="var(--text-muted)" stroke-width="1" opacity="0"/>
  </svg>
`;
const rateTooltip = document.createElement("div");
rateTooltip.className = "tooltip";
rateWrap.appendChild(rateTooltip);

const rateSvg = document.getElementById("rate-svg");
const rateHoverLine = rateSvg.querySelector(".hover-line");
rateSvg.addEventListener("mousemove", (evt) => {
  const p = toSvgPoint(rateSvg, evt);
  const frac = (p.x - RPAD_L) / (RW - RPAD_L - RPAD_R);
  let idx = Math.round(frac * (rateX.length - 1));
  idx = Math.max(0, Math.min(rateX.length - 1, idx));
  const xPix = xScaleRate(rateX[idx]);
  rateHoverLine.setAttribute("x1", xPix);
  rateHoverLine.setAttribute("x2", xPix);
  rateHoverLine.setAttribute("opacity", "1");
  const rect = rateSvg.getBoundingClientRect();
  const wrapRect = rateWrap.getBoundingClientRect();
  const px = rect.left + (xPix / RW) * rect.width;
  rateTooltip.style.left = (px - wrapRect.left) + "px";
  rateTooltip.style.top = (rect.top - wrapRect.top) + "px";
  rateTooltip.style.opacity = "1";
  const lines = DATA.slices.map(s => `${s}: ${DATA.rate_series[s][idx]}`).join("<br/>");
  rateTooltip.innerHTML = `step ${fmtInt(rateX[idx])}<br/>${lines}`;
});
rateSvg.addEventListener("mouseleave", () => {
  rateHoverLine.setAttribute("opacity", "0");
  rateTooltip.style.opacity = "0";
});

// ---------- histograms ----------
const histGrid = document.getElementById("hist-grid");
const HW = 320, HH = 160, HPAD_L = 30, HPAD_R = 8, HPAD_T = 8, HPAD_B = 22;

DATA.slices.forEach((s) => {
  const bins = DATA.size_bins[s];
  const sizes = Object.keys(bins).map(Number).sort((a, b) => a - b);
  const counts = sizes.map(sz => bins[sz]);
  const maxCount = Math.max(...counts) * 1.1;
  const xScale = scaleLinear(sizes[0] - 0.5, sizes[sizes.length - 1] + 0.5, HPAD_L, HW - HPAD_R);
  const yScale = scaleLinear(0, maxCount, HH - HPAD_B, HPAD_T);
  const barW = Math.min(24, (HW - HPAD_L - HPAD_R) / sizes.length - 2);

  const bars = sizes.map((sz, i) => {
    const cx = xScale(sz);
    const y = yScale(counts[i]);
    const h = (HH - HPAD_B) - y;
    return `<rect class="hist-bar" data-size="${sz}" data-count="${counts[i]}"
              x="${cx - barW/2}" y="${y}" width="${barW}" height="${h}"
              rx="3" fill="${COLORS[s]}"/>`;
  }).join("");

  const xLabels = sizes.filter((_, i) => i % Math.ceil(sizes.length / 6) === 0)
    .map(sz => `<text class="axis-text" x="${xScale(sz)}" y="${HH - 6}" text-anchor="middle">${sz}</text>`).join("");
  const yTicks = [0, Math.round(maxCount/2), Math.round(maxCount)];
  const yLines = yTicks.map(v => {
    const y = yScale(v);
    return `<line class="grid-line" x1="${HPAD_L}" x2="${HW-HPAD_R}" y1="${y}" y2="${y}"/>
            <text class="axis-text" x="2" y="${y+3}">${v >= 1000 ? Math.round(v/1000)+'k' : v}</text>`;
  }).join("");

  const col = document.createElement("div");
  col.className = "hist-col chart-wrap";
  col.innerHTML = `
    <div class="hist-title" style="color:${COLORS[s]}">${s}</div>
    <svg class="chart" viewBox="0 0 ${HW} ${HH}">
      ${yLines}
      ${bars}
      <line class="baseline" x1="${HPAD_L}" x2="${HW-HPAD_R}" y1="${HH-HPAD_B}" y2="${HH-HPAD_B}"/>
      ${xLabels}
    </svg>
  `;
  histGrid.appendChild(col);

  const tooltip = document.createElement("div");
  tooltip.className = "tooltip";
  col.appendChild(tooltip);
  const svg = col.querySelector("svg");
  svg.querySelectorAll(".hist-bar").forEach((bar) => {
    bar.addEventListener("mousemove", (evt) => {
      const rect = svg.getBoundingClientRect();
      const colRect = col.getBoundingClientRect();
      const barRect = bar.getBoundingClientRect();
      tooltip.style.left = (barRect.left + barRect.width/2 - colRect.left) + "px";
      tooltip.style.top = (barRect.top - colRect.top) + "px";
      tooltip.style.opacity = "1";
      tooltip.innerHTML = `size ${bar.dataset.size}: ${Number(bar.dataset.count).toLocaleString()}`;
    });
    bar.addEventListener("mouseleave", () => { tooltip.style.opacity = "0"; });
  });
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    run()
