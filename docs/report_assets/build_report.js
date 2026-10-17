// Builds JEPA/prototype_new/PROJECT_COMPREHENSIVE_REPORT.docx
// Run from JEPA/:  NODE_PATH=<dir with node_modules> node prototype_new/docs/report_assets/build_report.js
// Every numeric table is read from result files under prototype_new/results at build time.
const fs = require('fs');
const path = require('path');
const D = require('docx');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, ImageRun, Header, Footer, PageNumber, TableOfContents, LevelFormat, PageBreak,
} = D;

const ROOT = 'prototype_new';
const RES = path.join(ROOT, 'results');
const FIG = path.join(ROOT, 'docs/report_assets/figures');
const OUT = process.env.REPORT_OUT || path.join(ROOT, 'PROJECT_COMPREHENSIVE_REPORT.docx');
const J = (p) => JSON.parse(fs.readFileSync(path.join(RES, p), 'utf8').replace(/\bNaN\b/g, 'null'));
const f = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? '—' : Number(x).toFixed(d));
const sgn = (x, d = 1) => (x >= 0 ? '+' : '') + Number(x).toFixed(d);
const ci = (c, d = 1) => `[${f(c[0], d)}, ${f(c[1], d)}]`;
const pct = (x, d = 0) => `${(100 * x).toFixed(d)}%`;

// ------------------------------------------------------------------ evidence
const legacyDev = J('controlled_development/summary.json').methods;
const legacyFc = J('controlled_development/forecast_baseline.json');
const legacyAE = J('controlled_development/action_effects.json').rows;
const legacySmoke = J('controlled_smoke/summary.json').methods;
const legacyRetail = J('retail_audited_verified/metrics.json');
const trackA = J('isomorph_prediction/metrics.json');
const cov = Object.fromEntries(['v3a', 'v3b', 'v3c', 'v3d'].map((v) => [v, J(`decision_${v}/data/coverage_report.json`)]));
const models = J('decision_v3d/models/summary.json');
const ev = J('decision_v3d/evaluation/summary.json');
const retail = J('retail_attribution/metrics.json');
const retailMan = J('retail_attribution/data_manifest.json');
const planP = process.env.PLAN_SUMMARY || path.join(RES, 'decision_v3d/planning_test/summary.json');
const plan = fs.existsSync(planP) ? JSON.parse(fs.readFileSync(planP, 'utf8')) : null;
const valBS = (() => {
  const by = {};
  for (const l of fs.readFileSync(path.join(RES, 'decision_v3d/planning_validation_basestock/episodes.jsonl'), 'utf8').trim().split('\n')) {
    const r = JSON.parse(l); (by[r.method] = by[r.method] || []).push(r.cost);
  }
  return Object.fromEntries(Object.entries(by).map(([k, v]) => [k, v.reduce((a, b) => a + b, 0) / v.length]));
})();
const aeMean = (m, k) => { const r = legacyAE.filter((x) => x.method === m); return r.reduce((a, b) => a + b[k], 0) / r.length; };
const trackAmean = (m, h) => { const r = trackA.rows.filter((x) => x.method === m && x.horizon === h); return r.reduce((a, b) => a + b.rmse, 0) / r.length; };
const trackAwins = (h) => { let w = 0; for (let k = 0; k < 4; k++) { const p = trackA.rows.find((x) => x.rollout === k && x.method === 'persistence' && x.horizon === h); const r = trackA.rows.find((x) => x.rollout === k && x.method === 'ridge' && x.horizon === h); if (r.rmse < p.rmse) w++; } return w; };
const E = (split, m) => ev[split].methods[m];
const mono = (split, m, k) => { const a = E(split, m).mono_members; return a.reduce((s, x) => s + x[k], 0) / a.length; };
const P = (m) => (plan ? plan.methods[m] : null);

// ------------------------------------------------------------------ helpers
const FONT = 'Calibri';
const COLORS = { head: '1F3864', accent: '2E75B6', shade: 'DCE6F2', note: 'F2F2F2', warn: 'FBE5D6' };
const items = []; // {kind, ...}
const figs = {}; const tabs = {}; let figN = 0; let tabN = 0;

function runs(text, base = {}) {
  const out = [];
  const parts = String(text).split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/);
  for (const p of parts) {
    if (!p) continue;
    if (p.startsWith('`')) out.push(new TextRun({ text: p.slice(1, -1), font: 'Consolas', size: 19, color: '1F3864', ...base }));
    else if (p.startsWith('**')) out.push(new TextRun({ text: p.slice(2, -2), bold: true, ...base }));
    else if (p.startsWith('*') && p.length > 2) out.push(new TextRun({ text: p.slice(1, -1), italics: true, ...base }));
    else out.push(new TextRun({ text: p, ...base }));
  }
  return out;
}
const H1 = (t) => items.push({ k: 'h', level: HeadingLevel.HEADING_1, t, pb: true });
const H2 = (t) => items.push({ k: 'h', level: HeadingLevel.HEADING_2, t });
const H3 = (t) => items.push({ k: 'h', level: HeadingLevel.HEADING_3, t });
const Pa = (t) => items.push({ k: 'p', t });
const B = (...list) => list.forEach((t) => items.push({ k: 'b', t }));
const NOTE = (t, warn = false) => items.push({ k: 'note', t, warn });
const EQ = (t, label = '') => items.push({ k: 'eq', t, label });
const CODE = (...lines) => lines.forEach((t) => items.push({ k: 'code', t }));
const FIGURE = (key, file, caption, width = 6.3) => items.push({ k: 'fig', key, file, caption, width });
const TABLE = (key, caption, header, rows, widths) => items.push({ k: 'tab', key, caption, header, rows, widths });
const PART = (t) => items.push({ k: 'part', t });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function resolve(t) {
  return String(t).replace(/\[\[(fig|tab):([a-z0-9_]+)\]\]/g, (_, kind, key) => {
    const n = kind === 'fig' ? figs[key] : tabs[key];
    if (!n) throw new Error(`Unresolved reference ${kind}:${key}`);
    return `${kind === 'fig' ? 'Figure' : 'Table'} ${n}`;
  });
}
function cell(text, width, opts = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: opts.header ? { fill: COLORS.shade, type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ spacing: { before: 0, after: 0 }, children: runs(resolve(text), { size: 18, bold: !!opts.header }) })],
  });
}
const border = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };

function render() {
  // number figures/tables in document order first
  for (const it of items) {
    if (it.k === 'fig') figs[it.key] = ++figN;
    if (it.k === 'tab') tabs[it.key] = ++tabN;
  }
  const out = [];
  for (const it of items) {
    if (it.k === 'part') {
      out.push(new Paragraph({ children: [new PageBreak()] }));
      out.push(new Paragraph({ spacing: { before: 2400, after: 200 }, alignment: AlignmentType.LEFT,
        children: [new TextRun({ text: it.t, size: 40, bold: true, color: COLORS.accent })] }));
      out.push(new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: COLORS.accent, space: 1 } }, children: [] }));
    } else if (it.k === 'h') {
      out.push(new Paragraph({ heading: it.level, children: [new TextRun(it.t.replace(/`/g, ''))] }));
    } else if (it.k === 'p') {
      out.push(new Paragraph({ spacing: { after: 140, line: 288 }, children: runs(resolve(it.t)) }));
    } else if (it.k === 'b') {
      out.push(new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 60, line: 276 }, children: runs(resolve(it.t)) }));
    } else if (it.k === 'note') {
      out.push(new Paragraph({ spacing: { before: 120, after: 160, line: 276 }, indent: { left: 240, right: 240 },
        shading: { fill: it.warn ? COLORS.warn : COLORS.note, type: ShadingType.CLEAR, color: 'auto' },
        border: { left: { style: BorderStyle.SINGLE, size: 18, color: it.warn ? 'C55A11' : COLORS.accent, space: 8 } },
        children: runs(resolve(it.t), { size: 20 }) }));
    } else if (it.k === 'eq') {
      out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120 },
        children: [new TextRun({ text: it.t, font: 'Cambria Math', italics: true, size: 22 }),
          ...(it.label ? [new TextRun({ text: `    (${it.label})`, size: 20, color: '595959' })] : [])] }));
    } else if (it.k === 'code') {
      out.push(new Paragraph({ spacing: { before: 0, after: 0 }, indent: { left: 360 }, shading: { fill: 'F7F7F7', type: ShadingType.CLEAR, color: 'auto' },
        children: [new TextRun({ text: it.t, font: 'Consolas', size: 17 })] }));
    } else if (it.k === 'fig') {
      const file = path.join(FIG, it.file); const s = pngSize(file);
      const w = Math.round(it.width * 96); const h = Math.round(w * s.h / s.w);
      out.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 60 },
        children: [new ImageRun({ type: 'png', data: fs.readFileSync(file), transformation: { width: w, height: h },
          altText: { title: `Figure ${figs[it.key]}`, description: it.caption, name: it.file } })] }));
      out.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 220 }, indent: { left: 360, right: 360 },
        children: [new TextRun({ text: `Figure ${figs[it.key]}. `, bold: true, size: 18, color: COLORS.head }), ...runs(resolve(it.caption), { size: 18, color: '404040' })] }));
    } else if (it.k === 'tab') {
      const total = 9360; const ws = it.widths || it.header.map(() => Math.floor(total / it.header.length));
      const scale = total / ws.reduce((a, b) => a + b, 0); const W = ws.map((x) => Math.floor(x * scale));
      W[W.length - 1] += total - W.reduce((a, b) => a + b, 0);
      out.push(new Paragraph({ keepNext: true, spacing: { before: 200, after: 80 },
        children: [new TextRun({ text: `Table ${tabs[it.key]}. `, bold: true, size: 18, color: COLORS.head }), ...runs(resolve(it.caption), { size: 18, color: '404040' })] }));
      out.push(new Table({
        width: { size: total, type: WidthType.DXA }, columnWidths: W,
        borders: { top: border, bottom: border, left: border, right: border, insideHorizontal: border, insideVertical: border },
        rows: [new TableRow({ tableHeader: true, children: it.header.map((h, i) => cell(h, W[i], { header: true })) }),
          ...it.rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, W[i])) }))],
      }));
      out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
    }
  }
  return out;
}

// ================================================================== CONTENT
const today = '17 October 2026';

// ---------------- Part I
PART('Part I — Executive overview');
H1('1. About this report');
Pa('This report describes the `JEPA/prototype_new` subproject as it exists on ' + today + '. It is written for colleagues who have not followed the project. It explains the research question, the system that was built, the experiments that were run, what the evidence shows, and what remains unresolved. Every number in the results tables is read directly from result files in `prototype_new/results/` when the report is generated (`docs/report_assets/build_report.js`); figures are regenerated by `docs/report_assets/make_figures.py`.');
NOTE('Status: technical and experimental status report. Findings represent the audited state of the workspace on the report date. Simulator costs are assumed accounting units, not money. No result in this report demonstrates real-world operational effectiveness.', true);

H1('2. Executive summary');
Pa('**What the project is trying to do.** The project asks whether a learned, action-conditioned *world model* — a neural network that predicts how a supply chain will evolve if a particular decision is taken — can predict the consequences of operational decisions well enough to choose better decisions than simple rules. The design draws on the Joint-Embedding Predictive Architecture (JEPA) family, in which a network learns to predict an abstract (latent) representation of the future instead of every raw detail.');
Pa('**Why this could matter.** Supply-chain decisions are sequential: dispatching stock today changes what is available tomorrow, which changes tomorrow\'s options. A model that can imagine the effect of alternative decisions could be used by a planner to compare them before acting. That requires more than forecasting demand; it requires predicting how *actions* change inventory, backlog and cost.');
Pa('**What has been built.** The active implementation wraps the open-source ISOMORPH supply-chain simulator (MIT licence) with an adapter that lets an external controller choose per-item dispatch quantities to a single destination. Around it sit a data-generation pipeline, a coverage checker, a family of dynamics models (persistence, ridge regression, a supervised neural network, and one latent encoder–predictor–decoder architecture trained under four objectives including EMA-target and SIGReg variants), a paired counterfactual evaluation harness, a receding-horizon cross-entropy-method (CEM) planner, and a separate real-data track on the FreshRetailNet-50K retail dataset.');
Pa('**What the evidence says.** The earliest controlled experiments were negative: learned planners cost far more than the simulator\'s native rule (e.g. 8,893 vs 350 assumed units in the first smoke run), the training data contained no backlog events (0/288 transitions), and the native rule had an information-timing advantage. This iteration corrected the timing asymmetry, made counterfactual branches exactly paired, and regenerated training data until a pre-declared coverage gate passed. Under that corrected protocol:');
B(
  `A supervised direct-delta neural network (5-member ensemble) predicts paired one-step action effects better than a zero-change baseline on in-distribution test scenarios (error ${f(E('test', 'delta_mlp_ensemble').affected_mae.mean, 2)} vs ${f(E('test', 'persistence').affected_mae.mean, 2)} units, ${pct(E('test', 'delta_mlp_ensemble').paired.affected_mae_minus_persistence.rel)}; 95% CI of the difference excludes zero). It does **not** do so on shifted (out-of-distribution) scenarios.`,
  `Adding a JEPA-style latent objective (EMA target or SIGReg) did not improve physical prediction or decision ranking over the identical architecture trained without it; validation loss was worse (latent decoder-only ${f(models.latent_sup_0.validation, 3)}–${f(models.latent_sup_1.validation, 3)} vs EMA ${f(models.ema_0.validation, 3)}–${f(models.ema_2.validation, 3)} and SIGReg ${f(models.sigreg_2.validation, 3)}–${f(models.sigreg_0.validation, 3)}). The latent models also respond weakly to the action itself (they reproduce only 12–35% of the true response when a dispatch request is swept).`,
  plan ? `In receding-horizon planning on ${plan.episodes} paired test scenarios, all learned-model planners beat the simulator\'s native rule on average but none beat a simple base-stock rule whose single parameter was tuned on validation scenarios (best learned planner ${f(P('mpc_latent_sup_0').cost_mean)} vs base-stock ${f(P('basestock_k9').cost_mean)} assumed units; difference CI ${ci(P('mpc_latent_sup_0').minus_basestock_k9.ci95)}).` : 'Planning results were not available when this report was built.',
  `On FreshRetailNet (3,579 randomly sampled series from 289 stores), a JEPA-style EMA latent loss changed sales error by ${sgn(retail.paired_seed_mean_predictions.ema_jepa_minus_supervised.ae.mean, 4)} (CI ${ci(retail.paired_seed_mean_predictions.ema_jepa_minus_supervised.ae.ci95, 4)}) relative to the same network trained only on the supervised targets — no measurable benefit.`,
);
Pa('**Does the approach work?** Not yet, in the sense the project set out to test. The engineering foundation is now sound and a supervised dynamics model learns useful in-distribution action effects, but no evidence shows that a JEPA-style objective helps, learned models generalise poorly to shifted conditions, and model-based planning has not beaten a well-tuned simple rule. These are informative negative results rather than a failure of the infrastructure.');

H1('3. The project at a glance');
TABLE('glance', 'Project summary (verified against code and result files).', ['Dimension', 'Information'], [
  ['Research problem', 'Can a learned action-conditioned world model predict the consequences of supply-chain decisions well enough to improve goal-directed planning under uncertainty?'],
  ['Intended contribution', 'Determine whether JEPA-style latent prediction adds value over supervised dynamics for supply-chain decisions; produce an honest, reproducible evaluation protocol.'],
  ['Core architecture', 'Numeric state (214 features) → MLP encoder → latent predictor conditioned on dispatch actions → decoder to physical state change; compared with a direct supervised delta-MLP. `src/models/controlled.py`.'],
  ['Real-world dataset', 'FreshRetailNet-50K (Hugging Face revision c374e85…, CC BY 4.0): random 300-row-group sample, 3,579 series.'],
  ['Synthetic environment', 'ISOMORPH simulator, commit 9ab2689… (MIT), 13-node US network, 3-item catalogue, via `DecisionEnv`.'],
  ['World-model references', 'V-JEPA 2 / 2-AC (MIT), LeWorldModel (MIT), SupplyChain-JEPA (no open-source licence; concepts only).'],
  ['Planner', 'Seeded integer cross-entropy method, receding horizon (7-day plans, first action executed), `src/planning/cem.py`.'],
  ['Main baselines', 'Native ISOMORPH rule, validation-tuned base-stock rule, zero and random dispatch, persistence, ridge, ExtraTrees, simulator with forecast demand, exact-simulator oracle.'],
  ['Experiments completed', 'Legacy smoke/development planning; Track A native prediction; decision_v3 coverage, dynamics comparison, paired probes, planning; FreshRetailNet attribution (5 seeds).'],
  ['Strongest positive finding', 'Supervised delta-MLP ensemble beats zero-change on paired in-distribution action effects (−18%, CI excludes 0); its ensemble spread ranks its own errors (Spearman 0.58 iid, 0.84 OOD).'],
  ['Most important negative finding', 'No measurable benefit from JEPA-style latent objectives (simulator or retail); learned planners do not beat a tuned base-stock rule; action effects not learned out of distribution.'],
  ['Current research status', 'PARTIAL. Engineering and data gates pass; dynamics gate passes in-distribution only; planning-advantage gate not met; JEPA-benefit hypothesis not supported.'],
], [2200, 7160]);

// ---------------- Part II
PART('Part II — Problem formulation and motivation');
H1('4. The supply-chain decision problem');
Pa('Consider a retailer with one large city store (the *destination*) supplied by a network of warehouses. Each day customers arrive and buy what is on the shelf. If the shelf is empty, the unmet demand becomes *backlog* — orders owed to customers. Warehouses ship stock towards the store along road lanes that take one or more days and can carry only a limited volume each day. Warehouses themselves are refilled from upstream warehouses and ultimately from factories (*sources*), and those refills take an uncertain *lead time*.');
Pa('[[fig:network]] shows the actual network used in every controlled experiment in this project: the 13-node ISOMORPH topology with three sources, nine intermediate warehouses and a single destination (New York). Travel times on edges are in days. The two last-mile lanes into New York have limited daily capacity, which matters when large shipments are requested.');
FIGURE('network', 'f01_isomorph_network.png', 'The ISOMORPH network as instantiated by `build_example_simulation_from_adjacency` (vendored simulator). Nodes are placed at their real latitude/longitude; edge labels are travel times in days; red edges are the capacity-limited last-mile lanes. This is the only topology used in the controlled experiments.', 6.0);
Pa('The decision studied here is narrow and concrete: **each day, after observing the day\'s demand, how many units of each item should be dispatched from the warehouse network to the destination?** Dispatching too little causes backlog (expensive in the chosen accounting); dispatching too much moves stock unnecessarily, incurs transport cost and drains warehouses, which then trigger upstream replenishment. Decisions are sequential because today\'s dispatch arrives in one or more days and changes tomorrow\'s inventory, which changes tomorrow\'s best decision.');
Pa('It is useful to separate four tasks that are often conflated:');
B('**Predicting demand** — how many units customers will want tomorrow. This is a forecasting problem and does not involve decisions.',
  '**Predicting the future operational state** — inventory, backlog and in-transit stock tomorrow, under whatever policy is being followed.',
  '**Predicting the consequences of an action** — how tomorrow\'s state would differ if we dispatched 0 units versus 500 units today, all else equal. This is a counterfactual question.',
  '**Choosing an action** — using such predictions to pick the dispatch that best meets a goal (for example, lowest cost). This is planning.');
Pa('A good demand forecaster need not be a good action-effect predictor, and a good action-effect predictor need not yield a good planner. The project\'s central question lives at the third and fourth levels.');
H2('4.1 Original and current objectives');
Pa('The earlier project documents (`proposal.md`, `plan2.md`, the legacy `prototype/`) framed the goal as a risk-sensitive, decision-aligned JEPA world model choosing among five fixed action templates under disruption regimes, using a custom simulator. The current implementation plan (`round_claude/IMPLEMENTATION_PLAN.md`) reframes it as determining *whether* a learned action-conditioned model can predict consequences of feasible decisions well enough to improve goal-conditioned planning, with JEPA as one candidate rather than an assumed winner. The current code implements this narrower, testable version: a single controllable action type (dispatch requests) in a fixed network, with planning evaluated by realized simulator cost.');

H1('5. Why conventional approaches may be insufficient — and why they are the bar');
Pa('**Rule-based policies** such as base-stock (order-up-to) rules are simple, robust and often near-optimal for stationary demand. They do not reason about network interactions such as shared lane capacity or upstream replenishment. **Forecast-then-optimise** pipelines separate demand prediction from decision optimisation; they inherit forecast errors and usually require a hand-built model of the operational dynamics. **Direct supervised cost prediction** learns to score decisions but does not expose the intermediate states that explain *why* a decision is good. **Learned dynamics models** predict the next state from the current state and action and can be used with any planner, but errors compound over multi-step rollouts and planners can exploit model errors. **JEPA-style latent world models** predict a compressed representation of the future rather than every raw value, which in vision and robotics reduces the burden of predicting irrelevant detail.');
Pa('The motivation for this project is that supply-chain states are high-dimensional and partially relevant, so a latent model might focus on decision-relevant structure. This assumption is **untested** at the start of this work, and it is plausible that it does not hold for numeric state vectors where every feature is already meaningful. For that reason simple rules and supervised dynamics are treated as the bar to beat, not as straw men.');

H1('6. Research questions and hypotheses');
Pa('The implementation plan states seven hypotheses (H1–H7). The table records which were tested in the current workspace and with what outcome. The outcomes are this report\'s analytical reading of the evidence presented in Part VII.');
TABLE('hyp', 'Hypotheses from `round_claude/IMPLEMENTATION_PLAN.md` and their status.', ['Hypothesis', 'Tested by', 'Status on the report date'], [
  ['H1 Action-conditioned dynamics beat persistence and simple baselines', 'decision_v3 paired probes (P1, P2)', 'Supported in-distribution for the supervised delta-MLP; not supported on OOD scenarios.'],
  ['H2 Useful multi-step rollouts', 'P3 open-loop rollouts, horizons 1/3/7/14', 'Partially: learned models beat persistence on destination backlog and 7-day cost error; known-mechanics model is far better on pipeline state.'],
  ['H3 JEPA-style objective adds value over supervised dynamics', 'Four objectives on one architecture; retail attribution', 'Not supported (no improvement; some degradation).'],
  ['H4 Planning benefit vs fair baselines', 'Paired MPC on 60 test scenarios', 'Not supported against the tuned base-stock rule; learned planners beat the untuned native rule on average.'],
  ['H5 Uncertainty identifies unreliable predictions', 'Ensemble spread vs error; spread-penalised planner', 'Partially: spread ranks errors well; penalty did not significantly change planning cost.'],
  ['H6 Real-data relevance', 'FreshRetailNet attribution', 'Neural models competitive with ExtraTrees; no JEPA-specific benefit.'],
  ['H7 Generalisation', 'test_ood split (shifted demand, capacity, lead times, held-out behaviour)', 'Not supported: learned action-effect accuracy collapses to zero-change level.'],
], [3000, 2700, 3660]);

// ---------------- Part III
PART('Part III — JEPA and the world-model concept');
H1('7. JEPA explained for a new reader');
Pa('An **encoder** is a function that turns an observation into a vector of numbers, the **latent representation**. A **predictor** takes the latent representation of the present (and, in a world model, an action) and outputs a guess of the latent representation of the future. In a Joint-Embedding Predictive Architecture the training signal compares that guess with the encoding of the actual future, *in latent space*, rather than asking the network to reconstruct every raw input value.');
EQ('z_t = E_θ(s_t),    ẑ_{t+1} = P_φ(z_t, a_t),    L_latent = ‖ ẑ_{t+1} − sg( Ē(s_{t+1}) ) ‖²', '1');
Pa('Here s_t is the observed state at day t, a_t the action, E_θ the encoder with parameters θ, P_φ the predictor, Ē the *target encoder* and sg(·) the stop-gradient operator (the target is treated as a constant during back-propagation). If the encoder were trained only by this loss, it could trivially map every input to the same vector, making the loss zero: this is **representation collapse**. Two common remedies are used in the reference code:');
B('**EMA target encoder** (I-JEPA / V-JEPA stage-1 style): Ē is a slowly moving exponential moving average of the online encoder, Ē ← m·Ē + (1−m)·E with momentum m (0.99 here). The lag plus stop-gradient discourages collapse.',
  '**SIGReg** (LeWorldModel): no EMA; instead a regulariser pushes the distribution of encodings towards an isotropic Gaussian by comparing their empirical characteristic function along random projections with that of a standard normal (Epps–Pulley statistic).');
Pa('A JEPA objective learns a *representation*. It does not by itself produce inventory, backlog or cost numbers, and it says nothing about which representation is useful for decisions. A downstream decision requires either a decoder from latent to physical quantities, a cost head, or a goal expressed in the latent space.');
FIGURE('jepa', 'f02_jepa_concept.png', 'The latent world model as implemented in `LatentDynamics` (`src/models/controlled.py`). Solid parts exist in every variant; the target-encoder path and SIGReg term depend on the objective. In all physical-prediction variants a decoder maps consecutive latents to a change in the physical state, trained against the true change.');
H1('8. From JEPA to an action-conditioned world model');
Pa('An action-conditioned world model learns the mapping *current state + proposed action → predicted next state*. Applying it repeatedly to a proposed action sequence a_t, …, a_{t+H−1} yields an imagined trajectory, a **rollout**. A planner can roll out many candidate sequences, score each imagined trajectory with a cost function, and pick the best.');
EQ('ŝ_{t+k+1} = ŝ_{t+k} + Δ̂(ŝ_{t+k}, a_{t+k}),    Ĵ(a_{t:t+H}) = Σ_k c(ŝ_{t+k+1}, τ̂_{t+k})', '2');
Pa('In this project c(·) is a *known* cost function applied to predicted physical quantities (inventory, backlog) and a predicted transport volume τ̂. A latent rollout alone guarantees nothing about these quantities: a latent vector can be predicted accurately while the decoded inventory is wrong, or while the latent ignores the action entirely. That is why every model here is evaluated on decoded physical quantities and on paired action effects, not on latent loss. Planning additionally needs (i) feasible action projection, (ii) a cost on predicted states, and (iii) a search procedure; all three are implemented (Section 13).');
H1('9. Reference implementations and their influence');
Pa('Three local reference trees were audited. Their identities were verified from local source files (Section 16 and `docs/REFERENCE_AUDIT.md`). [[fig:refs]] contrasts them with this project.');
FIGURE('refs', 'f03_reference_comparison.png', 'Reference systems versus this project. The references differ substantially from one another; none is reproduced here. Licence status as verified from local files.');
H2('9.1 V-JEPA 2 and V-JEPA 2-AC');
Pa('V-JEPA 2 (Assran et al., 2025) trains a video Vision-Transformer encoder with a masked latent-prediction objective on internet-scale video. V-JEPA 2-AC post-trains an action-conditioned transformer predictor on robot trajectories. In the local trainer `vjepa2-main/app/vjepa_droid/train.py`, the action-conditioned stage computes both context and targets with a *frozen* pretrained encoder (no EMA update in that stage) and trains the predictor with an L1 teacher-forced loss plus an autoregressive rollout loss; 7-dimensional robot actions and states are embedded into the predictor (`src/models/ac_predictor.py`). Planning uses CEM to minimise the distance between a predicted and a goal embedding. **Adapted here:** the idea of an action-conditioned latent predictor rolled out over multiple steps. **Not implemented:** video encoders, pretrained weights, transformer predictors, goal-embedding planning. **Licence:** MIT (local `LICENSE`); no source copied. An earlier internal audit described the AC stage as using EMA targets; the local source shows a frozen encoder, and the audit was corrected.');
H2('9.2 LeWorldModel (local folder `le-wm-main`)');
Pa('The local folder named after the "LM-World model" is **LeWorldModel (LeWM)** by Maes, Le Lidec, Scieur, LeCun and Balestriero (arXiv 2603.19312, 2026). Identity was verified by byte-equality of `jepa.py`, `module.py`, `train.py` and `LICENSE` with upstream commit 8edfeb3… (recorded in `docs/REFERENCE_AUDIT.md`). LeWM trains a pixel encoder and an action-conditioned autoregressive predictor end-to-end with two terms: next-embedding MSE and SIGReg, without EMA or stop-gradient. **Adapted here:** the SIGReg statistic was re-implemented (`sigreg()` in `src/models/controlled.py`, tested against an independent formula) and used in the `sigreg` and `jepa_only` objectives. **Not implemented:** pixel encoders, transformer predictor, the external stable-worldmodel solver. **Licence:** MIT; attribution in `src/models/LEWM_LICENSE`.');
H2('9.3 SupplyChain-JEPA');
Pa('SupplyChain-JEPA (`reference_SupplyChain_JEPA`, clean Git checkout of github.com/htizhang/SupplyChain-JEPA at commit 214d115…) is a typed-token supply-chain world model with online and frozen/EMA target encoders and several optional losses (masked latent, variance/covariance, SIGReg, inverse action). Its own ISOMORPH adapter classifies native dispatch traces as non-interventional — a conclusion this project independently reached. **Its README grants no open-source licence**, so no code was copied; only concepts (typed alignment of state and action, train-only transforms, honest action-coverage accounting) influenced the design. A second copy, `SupplyChain_JEPA_outsourced/`, carries declared provenance that could not be verified as a Git commit; it is kept separate.');
TABLE('refcmp', 'Reference comparison.', ['Aspect', 'V-JEPA 2-AC', 'LeWorldModel', 'SupplyChain-JEPA', 'This project (v3)'], [
  ['Input', 'Video + robot state/actions', 'Pixels + actions', 'Typed numeric/categorical tokens', '214-D numeric post-demand state'],
  ['Encoder', 'ViT, frozen in AC stage', 'ViT trained end-to-end', 'Structured token encoder', 'MLP, trained from scratch'],
  ['Collapse handling', 'Frozen pretrained encoder', 'SIGReg', 'EMA/frozen target + options', 'Decoder supervision; ± EMA; ± SIGReg'],
  ['Action', '7-D continuous', 'Env. control vector', 'Action tokens', '3 integer dispatch requests'],
  ['Planner', 'CEM on goal-embedding distance', 'External CEM solver', 'Evaluation-time selection', 'Integer CEM on decoded known cost'],
  ['Code reuse here', 'None', 'SIGReg re-implemented (MIT)', 'None (no licence)', '—'],
], [1500, 1900, 1800, 2000, 2160]);

// ---------------- Part IV
PART('Part IV — System architecture');
H1('10. Overall architecture');
Pa('[[fig:arch]] shows the active runtime. The offline half generates action-labelled data from the simulator, checks it against a coverage gate and trains models. The online half is the receding-horizon loop used both by the evaluation harness and the planner. Legacy components (the first prototype\'s custom simulator, five fixed action templates and decision-aligned JEPA modules under `src/baselines`, `src/evaluation`, `src/losses`, `src/models/decision_aligned_jepa.py`) are retained for historical reproduction and are not in this runtime.');
FIGURE('arch', 'f04_system_architecture.png', 'Actual decision_v3 architecture. Top: data generation and training (`collect_decision_data.py`, `coverage_report.py`, `run_dynamics_v3.py`). Bottom: daily planning loop (`run_planning_v3.py` with `CEMPlanner`). Model checkpoints flow from training to planning; every executed action is a real simulator step.');
H1('11. Data and preprocessing pipeline');
Pa('Two completely separate data sources are used. The **simulator** supplies action-labelled, controllable trajectories for action-effect prediction and planning. **FreshRetailNet** supplies real observational retail histories for a forecasting comparison; it has no controllable actions and is never fed into the simulator.');
H2('11.1 Simulator data (decision_v3)');
Pa('Each transition stores the post-demand state, the executed dispatch request, the next post-demand state, realised dispatch, lane utilisation, transport/holding/backlog units, next-day demand and service, plus episode identifiers, scenario parameters and the full demand table (for diagnostics only). Behaviour policies only see the observation and their own random stream. Splits are by episode with disjoint seeds and configurations (checked by `validate_rollout_splits`). Normalisation statistics are computed on training transitions only and stored inside each checkpoint (`Normalizer`). [[fig:splits]] (left) shows the episode composition.');
H2('11.2 FreshRetailNet data');
Pa(`Inputs are the previous 14 days of observed sales and in-stock-hour counts per series; targets are next-day observed sales and a next-day stockout indicator (stock_hour6_22_cnt > 0). Target dates are chronological and disjoint: train ≤ 2024-05-31, validation 2024-06-01…06-12, test 2024-06-13…06-25 (${retailMan.split_counts.train.toLocaleString()} / ${retailMan.split_counts.validation.toLocaleString()} / ${retailMan.split_counts.test.toLocaleString()} windows). Input histories may reach back into earlier periods, which is legitimate past information. Feature scaling is fitted on training windows only. The official evaluation file is excluded.`);
FIGURE('splits', 'f07_splits.png', 'Left: decision_v3d episodes by split and behaviour family (each 60 days). The OOD split uses a held-out behaviour (`piecewise_constant`) and shifted scenario ranges. Right: FreshRetailNet chronological splits for the attribution run (counts of next-day windows).');
H1('12. Model architecture');
Pa('All dynamics models share one interface, `rollout(states [B,D], actions [B,H,A]) → (states [B,H,D], transport [B,H])`, returning physical units so that evaluation and planning treat them identically. Targets are per-feature normalised one-step changes Δs = s_{t+1} − s_t; 161 of 214 features change in the training data, the rest are held fixed. Inventory-type features are projected to be non-negative after every predicted step.');
TABLE('models', 'Dynamics comparators (`src/models/controlled.py`; trainable parameters from `results/decision_v3d/models/summary.json`).', ['Model', 'Mechanism', 'Training signal', 'Parameters', 'Known limitation'], [
  ['Persistence', 'ŝ_{t+1} = s_t', 'none', '0', 'Ignores actions by construction'],
  ['Train-mean delta', 'adds mean training change', 'none', '0', 'Ignores actions'],
  ['Ridge', 'linear in normalised (s, a)', 'closed form, α chosen on validation', String(models.ridge.params), 'Cannot represent truncation by stock/capacity'],
  ['Delta-MLP', '3×256 SiLU MLP on (s, a) → Δs, τ', '8-step open-loop rollout MSE', String(models.delta_mlp_0.params), 'Compounding error; OOD'],
  ['Delta-MLP ensemble', 'mean of 5 seeds', 'as above', `5 × ${models.delta_mlp_0.params}`, 'Spread not calibrated'],
  ['Latent (4 objectives)', 'encoder → residual latent predictor → decoder of (z_t, z_{t+1}) → Δs', 'see Section 15', String(models.latent_sup_0.params), 'Weak action sensitivity (Section 25)'],
], [1500, 2500, 2000, 1100, 2260]);
EQ('z_0 = E(norm(s_t)),   z_{k+1} = z_k + P([z_k, a_{t+k}/1000]),   Δ̂_k = Dec([z_k, z_{k+1}]) ⊙ σ_Δ', '3');
Pa('Equation 3 is the latent model\'s rollout: E, P and Dec are two-hidden-layer MLPs of width 256 with a 64-dimensional latent; σ_Δ is the per-feature training standard deviation of the change. Physical predictions accumulate decoded changes, so multi-step rollouts stay in latent space and never re-encode predicted states. The delta-MLP instead feeds its predicted physical state back as input at every step. Neither model is graph-structured.');
H1('13. Action space and decision-making interface');
Pa('The action is an integer vector a_t ∈ {0, …, 1000}³: one *dispatch request* per catalogue item. `DispatchSpace.project` rounds and clips; `validate` rejects anything else. The simulator then executes the request faithfully: it allocates units from warehouses in shortest-path order, limited by on-hand stock and by per-lane container capacity, so the realised dispatch can be smaller than requested. This is the only controllable decision. **Procurement quantities, supplier choice, routing, prioritisation and expediting are not controllable**; upstream (s,S) replenishment and source ordering remain the simulator\'s built-in rules.');
Pa('The first prototype offered five fixed action templates (`src/env/actions.py`); these are legacy baselines only. The current planner searches the full integer request space over a 7-day horizon.');
H2('13.1 The planner');
Pa('The cross-entropy method keeps a Gaussian over action sequences, samples 64 candidates, scores them by predicted 7-day cost, keeps the 8 best, refits mean and standard deviation, and repeats three times. The mean is warm-started at the native rule\'s current action, the zero sequence and the warm start are always evaluated, and all candidates are projected to valid integers. Only the first day of the best sequence is executed; the next day the loop replans from the newly observed state (receding-horizon model-predictive control). [[fig:planner]] summarises the loop.');
FIGURE('planner', 'f05_planner_loop.png', 'Receding-horizon CEM loop as implemented in `CEMPlanner.plan` and `run_planning_v3.py`. All planners receive the same budget (64 samples × 3 iterations per day).');
Pa('Known limitations: finite 7-day look-ahead (terminal state is charged in evaluation by a 14-day native-rule tail, not inside the planner); no hard service-level constraint (backlog is a cost penalty only); the uncertainty penalty is a heuristic (mean + 1 × ensemble standard deviation), not a calibrated risk measure.');
H1('14. Simulator and operational dynamics');
Pa('The upstream simulator (`src/vendor/isomorph/simulator.py`, from github.com/tuhinsahai/ISOMORPH at commit 9ab2689…, MIT) advances one day per step. Two local changes were made, both documented in `PROVENANCE.json` and covered by parity tests: an optional external dispatch override, and splitting `step()` into a pre-dispatch and a post-dispatch phase with unchanged order of operations. The adapter `DecisionEnv` (`src/env/isomorph.py`) adds:');
B('**Post-demand decision timing.** Arrivals and the day\'s demand are realised first; the controller then observes the state and acts. This is the information set the native rule uses, so all controllers are information-matched. The legacy adapter decided *before* demand, giving the native rule an advantage.',
  '**Common random numbers.** Demand is tabulated at reset (identical values to the upstream generator) and source lead-time draws are keyed by (day, node, item). Under the legacy sequential random stream, lead-time draws were consumed policy-dependently (1 vs 111 draws over 48 days for native vs random), so branches were not truly paired beyond one step.',
  '**Exact cloning** (`clone()`), so a branch can be replayed from any state with identical future randomness.',
  '**Observed lane capacity.** The upstream builder sizes last-mile capacity from the realised horizon-mean demand intensity of the 50-item catalogue. Across training episodes this correlates with future 3-item demand (Spearman 0.70) but adds little beyond today\'s demand (R² 0.729 → 0.765). It is exposed to every controller as known metadata.');
CODE('day t:  1 receive (s,S) orders  2 destination arrivals clear backlog  3 reset lane containers',
  '        4 demand realised; unmet demand → backlog   ← observation s_t, controller chooses a_t',
  '        5 dispatch up to a_t per item (stock- and capacity-limited)  6 warehouse (s,S) replenishment',
  '        7 source orders with keyed lead time   → day t+1 steps 1–4 → observation s_{t+1}');
Pa('Simplifications that bound interpretation: a single fixed topology; three items; sources replenish themselves after a stochastic lead time ("magic" factories); no explicit disruptions in the iid scenarios (demand bursts appear only in the OOD split); deterministic travel times.');
H1('15. Losses, objectives and cost accounting');
Pa('All physical-prediction objectives share the decoder loss on normalised changes over an 8-step open-loop rollout:');
EQ('L_dec = (1/H) Σ_k [ (1/|M|) Σ_{j∈M} ((ŝ_{k,j} − s_{k,j}) / σ_Δ,j)² + ((τ̂_k − τ_k)/σ_τ)² ]', '4');
Pa('M is the set of features whose change varies in training, τ the transport volume. The four latent objectives differ only in what is added:');
TABLE('objectives', 'Training objectives of the latent architecture (identical networks; λ = 1, β = 0.1).', ['Objective', 'Loss', 'What it tests'], [
  ['latent_sup', 'L_dec', 'Architecture control: latent bottleneck without any JEPA term'],
  ['ema', 'L_dec + λ‖ẑ_k − sg(Ē(s_k))‖², Ē = EMA(E), m = 0.99', 'EMA-target JEPA term (I-JEPA / V-JEPA stage-1 style)'],
  ['sigreg', 'L_dec + λ‖ẑ_k − E(s_k)‖² + β·SIGReg(E(s))', 'LeWM-style term without EMA'],
  ['jepa_only', 'λ‖ẑ_k − E(s_k)‖² + β·SIGReg; decoder trained on detached latents', 'Whether the JEPA objective alone retains action-relevant information'],
], [1500, 4300, 3560]);
Pa('The planner\'s objective is different from any training loss: it is the predicted 7-day sum of the *known* cost function, Equation 5, applied to predicted states. Evaluation uses the same function on realised states. The weights are **assumed accounting units chosen by the project**, not data supplied by ISOMORPH and not money.');
EQ('c_t = 0.001 · Σ inventory(all nodes) + 1.0 · Σ backlog(destination) + 0.01 · Σ shipped units × path edges', '5');
Pa('Because upstream holding is large and mostly unaffected by dispatch, the controllable part of cost is a modest share of the total; component breakdowns and a weight-sensitivity check are reported with the planning results.');

// ---------------- Part V
PART('Part V — Data sources');
H1('16. ISOMORPH');
Pa('ISOMORPH is a published digital-twin supply-chain simulator (github.com/tuhinsahai/ISOMORPH, MIT) with a released dataset on Hugging Face (`Isomorph2026/isomorph-supply-chain-benchmark`, CC BY 4.0, pinned revision 478ee4d…). The dataset card describes 49 rollouts (2 baselines, 27 mixture scenarios, 20 Latin-hypercube uncertainty perturbations), all with seed 2025 and horizon 52,560 days on the same 13-node topology; the local manifest (`results/execution_audit/isomorph_manifest.csv`) lists 49 rows consistent with this. The full release (~99 GB, a card estimate) was **not** downloaded. The pinned simulator README defaults to 7,300-day runs; the exact mapping from the pinned source commit to the released trajectories remains unresolved.');
Pa('The released trajectories record the built-in policy\'s *outcomes* (shipments), not externally chosen actions, so they cannot directly answer "what if we had dispatched differently". All controlled data in this project is therefore generated locally with `DecisionEnv`. Default-policy parity with the unmodified upstream file was verified for two fixtures (seed 5, 50 items, 8 days; seed 19, 3 items, 32 days), and the native rule applied through the external interface reproduces the upstream built-in rule exactly (`tests/test_decision_env.py`).');
H2('16.1 Locally generated data and the coverage gate');
Pa('Earlier controlled data had essentially no shortage events (0 of 288 training transitions with positive backlog in the first smoke run). For this iteration, a coverage gate was fixed before any model was trained, and data was regenerated until it passed. Three versions failed and were preserved; the fourth (v3d, 400/60/60/48 episodes, seeds 34000–34567) passed all eight checks ([[fig:coverage]]).');
FIGURE('coverage', 'f08_coverage.png', 'Coverage-gate metrics for training data versions v3a–v3d (`results/decision_v3*/data/coverage_report.json`). Only behaviour policies and training size changed between versions; no model was trained until v3d.');
TABLE('cov', 'Training-split coverage by data version.', ['Version', 'Train transitions', 'Backlog > 0', 'Truncated requests', 'Lane saturated', 'Recovery events', 'Gate'], ['v3a', 'v3b', 'v3c', 'v3d'].map((v) => {
  const t = cov[v].splits.train; return [v, String(t.transitions), pct(t.backlog_positive_frac, 1), pct(t.truncated_request_frac, 1), pct(t.capacity_saturated_frac, 1), String(t.recovery_events), cov[v].coverage_gate_pass ? 'pass' : 'fail'];
}), [900, 1500, 1300, 1500, 1400, 1400, 1360]);
H1('17. FreshRetailNet-50K');
Pa('FreshRetailNet-50K (Wang et al., 2025; Hugging Face `Dingdong-Inc/FreshRetailNet-50K`, CC BY 4.0, pinned revision c374e85…) contains daily and hourly sales and stock-status records for 50,000 store–product series of fresh goods, with weather, discount and holiday covariates. Its key property is that sales are **censored by stockouts**: when a product is out of stock, observed sales understate true demand. This project predicts *observed* sales and a stockout indicator; it does not recover latent demand.');
Pa(`The historical local subset (first 24 row groups) held 267 series from only 3 stores. For this report\'s attribution run, 300 of 4,500 official-train row groups were sampled uniformly (seed 6101) with bounded, checksummed HTTP range reads: ${retailMan.series.toLocaleString()} series, ${retailMan.stores} stores, ${retailMan.products} products, 300,000 rows. Row groups are entity clusters, so this is a cluster sample, not a simple random sample of series. Stockout prevalence on test target days is ${pct(retailMan.stockout_rate.test, 1)}.`);
H1('18. Other datasets and environments');
TABLE('envs', 'Environment and dataset comparison.', ['Source', 'Realism', 'Controllable action', 'Network', 'Counterfactuals', 'Use here'], [
  ['ISOMORPH simulator via DecisionEnv', 'Synthetic, published model', 'Dispatch requests only', '13-node multi-echelon', 'Exact (clone + CRN)', 'Primary controlled environment'],
  ['ISOMORPH released trajectories', 'Synthetic', 'None (outcomes only)', '13-node', 'No', 'Metadata audited; not downloaded'],
  ['FreshRetailNet-50K', 'Real retail', 'None', 'None (store–product)', 'No', 'Observational prediction track'],
  ['OR-Gym inventory environments', 'Synthetic', 'Order quantities per stage', 'Serial/network', 'Possible', 'Considered; not installed or evaluated'],
  ['Legacy custom simulator (`src/env/simulator.py`)', 'Synthetic, ad hoc', 'Five templates', 'Small', 'CRN branching', 'Historical only'],
  ['DataCo Smart Supply Chain', 'Real orders/shipping', 'None', 'Partial', 'No', 'Listed in plan; not investigated'],
], [2100, 1300, 1500, 1300, 1400, 1760]);

// ---------------- Part VI
PART('Part VI — Training and experimental methodology');
H1('19. Training procedure');
Pa('Training windows are contiguous 8-day slices from training episodes (21,200 windows; 3,180 validation windows). Models are trained with Adam (learning rate 1e-3, batch 256, gradient clipping at 10) for 60 epochs; after every epoch the 8-step validation loss (Equation 4) is computed, and the best epoch\'s weights are kept. Test data is never used for selection. The EMA target updates after every optimiser step with momentum 0.99. Ridge chooses α from {0.01 … 1000} by validation loss (selected α = ' + models.ridge.alpha + '). Seeds: delta-MLP 0–4; each latent objective 0–2. Configuration: `configs/dynamics_v3.json`.');
TABLE('val', 'Best validation loss (normalised 8-step rollout error; lower is better).', ['Model', 'Seeds', 'Validation loss', 'Best epochs'], [
  ['persistence', '—', f(models.persistence.validation, 3), '—'],
  ['ridge', '—', f(models.ridge.validation, 3), '—'],
  ...[['delta_mlp', [0, 1, 2, 3, 4]], ['latent_sup', [0, 1, 2]], ['ema', [0, 1, 2]], ['sigreg', [0, 1, 2]], ['jepa_only', [0, 1, 2]]].map(([m, s]) => [m, s.join(', '),
    s.map((i) => f(models[`${m}_${i}`].validation, 3)).join(' / '), s.map((i) => models[`${m}_${i}`].best_epoch).join(' / ')]),
], [2000, 1500, 3500, 2360]);
H1('20. Experimental tracks');
Pa('[[fig:tracks]] separates the evidence streams. They use incompatible protocols and must not be pooled.');
FIGURE('tracks', 'f15_tracks.png', 'Experimental tracks. Track B (legacy) used pre-demand timing and sequential random numbers and is superseded by Track B (v3) for all current claims; its results are kept for the audit trail.');
B('**Track A — native trajectory prediction** (`run_isomorph_prediction.py`): seeds 4701–4720, 12/4/4 native-policy rollouts of 40 days, persistence and ridge at 1/3/8 steps. No actions vary, so it cannot test action effects.',
  '**Track B (legacy)** (`run_controlled.py`): smoke (seed 1701, 24 days, 4 test episodes) and development (seed 2701, 48 days, 8 test episodes) planning runs with EMA, SIGReg, supervised and action-blind latent models; plus a one-step zero-vs-max action-effect probe.',
  '**Track B (v3)**: decision_v3d data; dynamics comparison with pre-registered estimands P1–P3 on 60 iid and 48 OOD test episodes; paired MPC planning on 60 test scenarios.',
  '**Track C — FreshRetailNet**: the earlier single-seed run on 256 convenience series (`retail_audited_verified`), and the new attribution run on the random sample (`retail_attribution`).');
H1('21. Baselines and fair comparison');
TABLE('baselines', 'Decision baselines in v3 planning. All observe the same post-demand state.', ['Baseline', 'What it does', 'What it tests'], [
  ['native', 'ISOMORPH built-in rule: dispatch ⌈backlog + 7·EMA − in-transit − on-hand⌉', 'The simulator designer\'s policy, now information-matched'],
  ['basestock_k9', 'Same order-up-to form with coverage k = 9, chosen among k ∈ {3…12} on 60 validation scenarios', 'A strong, tuned simple rule'],
  ['zero / random', 'No dispatch / uniform random requests', 'Floor and sanity checks'],
  ['mpc_<model>', 'CEM planner using a learned model and the known cost', 'H4 for each model'],
  ['mpc_sim_forecast', 'CEM with exact simulator mechanics but EMA demand forecast and mean lead times', 'What known structure + naive forecast achieves'],
  ['mpc_sim_oracle', 'CEM with an exact clone (true future demand and lead times)', 'Diagnostic ceiling; not deployable'],
], [1900, 4300, 3160]);
NOTE('Fairness notes. The legacy comparison was not information-matched: the native rule saw same-day demand while learned planners did not; a separately added matched forecast controller cost 690.35 vs native 697.18 there. In v3 all controllers share post-demand information. The v3 planner\'s CEM is warm-started at the native action, which biases planners towards the native rule; the base-stock k was tuned on validation scenarios, whereas planners received no equivalent tuning of horizon or budget.');
H1('22. Evaluation metrics');
TABLE('metrics', 'Metric definitions.', ['Metric', 'Definition', 'Direction / units', 'Cannot establish'], [
  ['P1 action-effect error', 'For each paired probe, (branch(a) − branch(0)) predicted vs realised one-step state change; MAE over destination inventory, backlog, pipeline and last-mile warehouse stock', 'Lower; units of stock', 'Multi-step or decision quality'],
  ['Sign agreement', 'Share of non-zero true changes whose predicted sign matches', 'Higher; fraction', 'Magnitude accuracy'],
  ['P2 Spearman / regret', 'Rank correlation of predicted vs realised 7-day cost across 8 candidate first actions; regret = realised cost of the model\'s choice − best candidate', 'Higher / lower; assumed units', 'Performance of a full planner'],
  ['P3 rollout MAE', 'Open-loop error per feature group at horizon h given logged actions', 'Lower; units', 'Counterfactual validity'],
  ['Action sensitivity ratio', 'Predicted / true range of item-1 pipeline response when sweeping its request 0→1000', '1 is correct', 'Accuracy at a specific action'],
  ['Mean realised cost', 'Sum of Equation 5 over 60 controlled + 14 tail days', 'Lower; assumed units', 'Real monetary value'],
  ['Same-day fill rate', 'Units served from stock / units demanded, controlled days', 'Higher; fraction', 'Late service, backlog duration'],
  ['Sales MAE / WAPE', 'Mean |error|; Σ|error| / Σ observed sales', 'Lower; sales units / ratio', 'True (uncensored) demand accuracy'],
  ['Stockout AUROC / AUPRC / Brier', 'Ranking and probability quality for next-day stockout', 'Higher / higher / lower', 'Decision value'],
], [1900, 3700, 1700, 2060]);
Pa('Uncertainty: CIs are 95% bootstrap intervals over *episodes* (simulator) or *entities* (retail), never over time steps, which are not independent. Practical-relevance rule (pre-registered): a model beats a comparator only if the CI of the paired difference excludes zero and the improvement is at least 10%.');

// ---------------- Part VII
PART('Part VII — Results and interpretation');
H1('23. Results dashboard');
TABLE('dash', 'Key results, each with its own protocol (do not compare across rows).', ['Result', 'Protocol', 'Value', 'Evidence class'], [
  ['Legacy learned planners vs native', 'Track B legacy development, 8 episodes', `native ${f(legacyDev.native_policy.cost_mean)} vs best learned (SIGReg) ${f(legacyDev.sigreg_jepa.cost_mean)}`, 'Development, superseded'],
  ['Legacy action-effect probe', '8 paired probes, 1 member', `zero-change ${f(aeMean('no_action', 'delta_mae'), 2)} vs EMA ${f(aeMean('ema_jepa', 'delta_mae'), 2)} mixed-unit MAE`, 'Diagnostic, reproduced'],
  ['P1 delta-MLP ensemble vs zero-change', 'v3 iid test, 60 episodes', `${pct(E('test', 'delta_mlp_ensemble').paired.affected_mae_minus_persistence.rel)} (CI ${ci(E('test', 'delta_mlp_ensemble').paired.affected_mae_minus_persistence.ci95, 2)} units)`, 'Pre-registered test'],
  ['P1 on OOD', 'v3 OOD test, 48 episodes', `${pct(E('test_ood', 'delta_mlp_ensemble').paired.affected_mae_minus_persistence.rel)} (CI ${ci(E('test_ood', 'delta_mlp_ensemble').paired.affected_mae_minus_persistence.ci95, 2)})`, 'Pre-registered test'],
  ['Best learned planner vs tuned base-stock', plan ? `v3, ${plan.episodes} paired scenarios` : 'pending', plan ? `${sgn(P('mpc_latent_sup_0').minus_basestock_k9.mean)} (CI ${ci(P('mpc_latent_sup_0').minus_basestock_k9.ci95)})` : '—', 'Test, single run'],
  ['EMA-JEPA vs supervised (retail MAE)', '42,769 windows, 5 seeds', `${sgn(retail.paired_seed_mean_predictions.ema_jepa_minus_supervised.ae.mean, 4)} (CI ${ci(retail.paired_seed_mean_predictions.ema_jepa_minus_supervised.ae.ci95, 4)})`, 'Frozen chronological test'],
], [2500, 2300, 2900, 1660]);
H1('24. Controlled planning results');
H2('24.1 Legacy development run (superseded protocol)');
Pa(`Under the legacy pre-demand protocol (48 days, 8 test episodes), the native rule cost ${f(legacyDev.native_policy.cost_mean)} and a matched-information forecast controller ${f(legacyFc.cost_mean)}; learned planners cost ${f(legacyDev.sigreg_jepa.cost_mean)} (SIGReg), ${f(legacyDev.supervised.cost_mean)} (supervised) and ${f(legacyDev.ema_jepa.cost_mean)} (EMA), with wide bootstrap intervals ([[fig:legacy]]). In the first smoke run all three learned planners produced the *identical* cost ${f(legacySmoke.ema_jepa.cost_mean, 2)} vs native ${f(legacySmoke.native_policy.cost_mean, 2)}.`);
FIGURE('legacy', 'f09_legacy_planning.png', 'Legacy development planning (`results/controlled_development/summary.json` and `forecast_baseline.json`), 8 test episodes, 95% bootstrap CIs where recorded. Purple bars are learned-model planners.', 6.0);
NOTE('Labelling discrepancy found during this audit: the earlier report called one method "no-action". In `run_controlled.py` it is a CEM planner driven by a model trained with actions zeroed out (an action-blind model), not a zero-dispatch policy. With constant predictions, CEM keeps its initial mean (about 500 units/day), which explains its 100% fill rate. A genuine zero-dispatch policy is catastrophic (v3: mean cost about 204,000).', true);
Pa('Why the learned planners lost in that run is now attributable to several measured causes rather than to "JEPA" per se: training data with almost no shortage events; the timing advantage of the native rule; mixed-unit, absolute-state targets that let large upstream features dominate the loss; and short training. [[fig:failure]] links these causes to outcomes only where the link was measured.');
FIGURE('failure', 'f16_failure_analysis.png', 'Failure analysis restricted to measured relationships. Each red box is a cause documented in this report; amber boxes are measured symptoms.');
H2('24.2 decision_v3 planning (current protocol)');
if (plan) {
  const order = Object.keys(plan.methods).sort((a, b) => plan.methods[a].cost_mean - plan.methods[b].cost_mean);
  TABLE('plan', `v3 planning on ${plan.episodes} paired test scenarios (same seeds, demand and lead times for every method). Cost: 60 controlled + 14 native-tail days, assumed units. Δ columns: paired mean difference with 95% bootstrap CI over episodes.`,
    ['Method', 'Mean cost [95% CI]', 'Δ vs base-stock k9', 'Δ vs native', 'Fill', 'Backlog days'],
    order.map((m) => { const e = plan.methods[m]; const b = e.minus_basestock_k9; const n = e.minus_native;
      return [m, `${f(e.cost_mean)} ${ci(e.cost_ci95)}`, b ? `${sgn(b.mean)} ${ci(b.ci95)}` : '—', n ? `${sgn(n.mean)} ${ci(n.ci95)}` : '—', f(e.fill_rate, 4), f(e.backlog_days, 1)]; }),
    [2300, 2000, 1800, 1800, 700, 760]);
  FIGURE('plan', 'f14_v3_planning.png', 'v3 planning costs (log scale) with 95% bootstrap CIs; zero dispatch omitted (mean ≈ 204,000). Purple: learned-model planners; green: known mechanics with forecast demand; amber: exact-simulator oracle (not deployable).', 6.0);
  const best = ['mpc_latent_sup_0', 'mpc_ema_0', 'mpc_delta_mlp_ensemble', 'mpc_delta_mlp_ensemble_unc', 'mpc_sigreg_0', 'mpc_delta_mlp_0'];
  Pa(`Validation tuning selected base-stock coverage k = 9 (validation mean ${f(valBS.basestock_k9)} vs native ${f(valBS.native)}). On test, base-stock k9 cost ${f(P('basestock_k9').cost_mean)} and the native rule ${f(P('native').cost_mean)}. Every learned-model planner beat the native rule on average, but none beat base-stock: differences ranged from ${sgn(Math.min(...best.map((m) => P(m).minus_basestock_k9.mean)))} to ${sgn(Math.max(...best.map((m) => P(m).minus_basestock_k9.mean)))} units, and base-stock had the lower cost in ${pct(1 - Math.max(...best.map((m) => P(m).minus_basestock_k9.win_rate)))}–${pct(1 - Math.min(...best.map((m) => P(m).minus_basestock_k9.win_rate)))} of episodes.`);
  if (!(P('mpc_sim_forecast') && P('mpc_sim_oracle'))) {
    NOTE('The simulator-reference planners (mpc_sim_forecast, mpc_sim_oracle) were still running when this report was built and are excluded from this table and figure; their results will appear in results/decision_v3d/planning_test/ when complete.', true);
  } else {
    Pa(`The known-mechanics planner with naive EMA demand forecasts cost ${f(P('mpc_sim_forecast').cost_mean)} (Δ vs base-stock ${sgn(P('mpc_sim_forecast').minus_basestock_k9.mean)} ${ci(P('mpc_sim_forecast').minus_basestock_k9.ci95)}), and the exact-simulator oracle ${f(P('mpc_sim_oracle').cost_mean)} (Δ ${sgn(P('mpc_sim_oracle').minus_basestock_k9.mean)} ${ci(P('mpc_sim_oracle').minus_basestock_k9.ci95)}). The oracle bounds what any model could achieve with this planner and budget; the gap between it and the learned planners is the room that model error and search leave on the table.`);
  }
  Pa('Interpretation. Learned planners improve on their warm start (the native rule) mostly by reducing backlog, but a single well-chosen coverage parameter does at least as well. The latent decoder-only model\'s good showing should not be read as evidence of better dynamics: its action sensitivity is the weakest (Section 25), so its plans stay close to the warm start plus search noise. The uncertainty penalty changed the ensemble planner\'s cost by ' + sgn(P('mpc_delta_mlp_ensemble_unc').cost_mean - P('mpc_delta_mlp_ensemble').cost_mean) + ' units, which is within noise.');
  TABLE('sens', 'Cost-weight sensitivity: the executed trajectories re-weighted (not re-optimised).', ['Method', 'default', 'holding ×10', 'backlog ×5', 'transport ×5'],
    order.filter((m) => m !== 'zero').map((m) => [m, ...['default', 'holding_x10', 'backlog_x5', 'transport_x5'].map((k) => f(plan.methods[m].sensitivity[k]))]), [2600, 1690, 1690, 1690, 1690]);
} else {
  Pa('Planning results were not available when this report was built.');
}
H1('25. Counterfactual action-effect results');
Pa(`**Legacy probe.** Eight test scenarios, 12 zero-dispatch days, then zero vs maximum dispatch; one frozen model member. The action-blind (zero-change) model had lower mixed-unit delta MAE (${f(aeMean('no_action', 'delta_mae'), 2)}) than EMA (${f(aeMean('ema_jepa', 'delta_mae'), 2)}), supervised (${f(aeMean('supervised', 'delta_mae'), 2)}) and SIGReg (${f(aeMean('sigreg_jepa', 'delta_mae'), 2)}). This was reproduced exactly from the saved checkpoints. However, the learned models\' predicted changes had cosine similarity 0.6–0.77 (EMA, supervised) with the true change: they captured direction but spread error over ~200 mostly-unaffected features, which a mixed-unit MAE penalises. The probe was one-step only and its metric conflated features of different scales.`);
Pa('**v3 probes (P1, P2).** From states at days 8, 20, 32 and 44 of each replayed test episode, eight candidate first actions were branched with identical randomness and a common continuation. [[fig:branches]] shows one real example: the delta-MLP ensemble orders the branches correctly but under-predicts the effect of the largest request.');
FIGURE('branches', 'f06_paired_branches.png', 'Real paired branches from test episode test_0, day 20 (solid: simulator; dashed: delta-MLP ensemble). Each branch differs only in the first-day dispatch; days 2–7 use the logged actions.', 6.0);
TABLE('p1', 'Pre-registered P1/P2 results (seed means; ± = SD across training seeds). Δ vs zero: paired difference in affected-feature MAE with 95% episode-bootstrap CI.', ['Model', 'P1 MAE iid', 'Δ vs zero iid', 'Sign agree', 'P2 Spearman', 'P2 regret', 'P1 MAE OOD', 'Δ vs zero OOD'],
  ['persistence', 'ridge', 'delta_mlp', 'delta_mlp_ensemble', 'latent_sup', 'ema', 'sigreg', 'jepa_only', 'sim_forecast'].map((m) => {
    const a = E('test', m); const o = E('test_ood', m); const pa = a.paired.affected_mae_minus_persistence; const po = o.paired.affected_mae_minus_persistence;
    const sd = (x) => (x.seed_sd ? ` ±${f(x.seed_sd, 2)}` : '');
    return [m, f(a.affected_mae.mean, 2) + sd(a.affected_mae), pa ? `${pct(pa.rel)} ${ci(pa.ci95, 2)}` : '—', f(a.sign_agree_nonzero.mean, 2), f(a.spearman.mean, 2), f(a.regret.mean, 1), f(o.affected_mae.mean, 2), po ? `${pct(po.rel)} ${ci(po.ci95, 2)}` : '—'];
  }), [1500, 1100, 1500, 800, 900, 900, 1000, 1660]);
FIGURE('p1', 'f10_action_effect.png', 'P1 one-step paired action-effect error on iid (left) and OOD (right) test episodes. Dashed line: zero-change predictor. Error bars: 95% CI of the paired difference from zero-change, drawn around each mean.');
Pa('**Reading the results.** In distribution, the supervised delta-MLP (single seeds −13%, ensemble −18%) and the latent models with decoder supervision (−11% to −14%) beat zero-change with CIs that exclude zero. `jepa_only` does not. Out of distribution, no learned model beats zero-change and ridge is markedly worse. The known-mechanics reference is essentially exact for one-step effects, confirming that one-step effects in this simulator are deterministic mechanics.');
Pa('P1 accuracy alone hides a crucial weakness. [[fig:sensitivity]] sweeps one item\'s request from 0 to 1000 and compares the range of the predicted response with the true range. The delta-MLP reproduces about ' + pct(mono('test', 'delta_mlp', 'range_ratio')) + ' of the true response; the latent models only ' + pct(mono('test', 'ema', 'range_ratio')) + ' (EMA), ' + pct(mono('test', 'latent_sup', 'range_ratio')) + ' (decoder-only) and ' + pct(mono('test', 'sigreg', 'range_ratio')) + ' (SIGReg); `jepa_only` and ridge overshoot several-fold. On OOD the EMA and decoder-only latent models are fully insensitive. Their P1 gains therefore come mostly from predicting action-independent changes, not from modelling the action. This distinguishes *poor action conditioning* from *poor representation*: the latent models represent the state well enough to reduce one-step error, yet barely encode the effect of the decision.');
FIGURE('sensitivity', 'f11_action_sensitivity.png', 'Predicted vs true response magnitude to a request sweep (log scale; 1 = correct). Values below 1 mean the model under-reacts to the action.', 6.0);
H1('26. Multi-step and native trajectory prediction');
Pa(`**Track A (legacy).** On 4 native-policy test rollouts, ridge had lower mixed-unit RMSE than persistence in ${trackAwins(1)}/4, ${trackAwins(3)}/4 and ${trackAwins(8)}/4 rollouts at horizons 1, 3 and 8 (means ${f(trackAmean('ridge', 1), 1)} vs ${f(trackAmean('persistence', 1), 1)}, ${f(trackAmean('ridge', 3), 1)} vs ${f(trackAmean('persistence', 3), 1)}, ${f(trackAmean('ridge', 8), 1)} vs ${f(trackAmean('persistence', 8), 1)}). The RMSE mixes incompatible units and is diagnostic only; the trajectories contain no action variation.`);
Pa(`**v3 P3.** On logged test actions ([[fig:rollout]], right), learned models clearly beat persistence on destination backlog at 7 days (ensemble ${f(E('test', 'delta_mlp_ensemble').dest_backlog_h7.mean, 1)} vs ${f(E('test', 'persistence').dest_backlog_h7.mean, 1)} units) and on 7-day cost error (${f(E('test', 'delta_mlp_ensemble').cost7_abs_err.mean, 1)} vs ${f(E('test', 'persistence').cost7_abs_err.mean, 1)}). For pipeline state, ridge (${f(E('test', 'ridge').pipeline_h7.mean, 1)}) beats the neural models (${f(E('test', 'delta_mlp').pipeline_h7.mean, 1)}), and known mechanics with forecast demand is far better (${f(E('test', 'sim_forecast').pipeline_h7.mean, 2)}) — while being *worse* on backlog (${f(E('test', 'sim_forecast').dest_backlog_h7.mean, 1)}) because its constant EMA forecast lags demand. Decision ranking (P2, left) follows: the forecast-mechanics model has the lowest regret (${f(E('test', 'sim_forecast').regret.mean, 1)}), the delta-MLP ensemble ${f(E('test', 'delta_mlp_ensemble').regret.mean, 1)}, and the JEPA-objective variants are worse (${f(E('test', 'ema').regret.mean, 1)}–${f(E('test', 'jepa_only').regret.mean, 1)}).`);
FIGURE('rollout', 'f12_ranking_rollout.png', 'Left: P2 regret of the first action chosen by each model among 8 paired candidates (iid test). Right: P3 open-loop destination-backlog error by horizon.');
Pa(`**Uncertainty (H5).** For the delta-MLP ensemble, the standard deviation of members\' predicted 7-day cost has Spearman correlation ${f(E('test', 'delta_mlp_ensemble').uncertainty.spearman_std_vs_abs_err, 2)} (iid) and ${f(E('test_ood', 'delta_mlp_ensemble').uncertainty.spearman_std_vs_abs_err, 2)} (OOD) with the absolute cost error; errors in the top-20% spread bucket average ${f(E('test', 'delta_mlp_ensemble').uncertainty.err_top20pct_std, 0)} vs ${f(E('test', 'delta_mlp_ensemble').uncertainty.err_rest, 0)} units elsewhere (iid). Spread is a useful *ranking* of unreliability; it is not calibrated as a probability.`);
H1('27. FreshRetailNet results');
Pa(`**Earlier single-seed run** (256 convenience series, 3,328 test windows): last value WAPE ${f(legacyRetail.last_value.wape, 4)} / AUROC ${f(legacyRetail.last_value.auroc, 4)}; ExtraTrees ${f(legacyRetail.extratrees.wape, 4)} / ${f(legacyRetail.extratrees.auroc, 4)}; EMA latent+supervised ${f(legacyRetail.ema_latent_supervised.wape, 4)} / ${f(legacyRetail.ema_latent_supervised.auroc, 4)}. That run had no ablation isolating the latent loss and no uncertainty estimate.`);
Pa('**Attribution run** (random sample, 5 seeds, entity-clustered CIs; [[fig:retail]]). The same encoder–predictor–heads network was trained four ways. With the EMA latent loss it is indistinguishable from training on the supervised targets alone; a SIGReg-pretrained frozen representation is better than a random frozen one but worse than supervised training; all neural variants trained end-to-end slightly beat ExtraTrees on MAE.');
TABLE('retail', `FreshRetailNet attribution test results (${retailMan.split_counts.test.toLocaleString()} windows, ${retailMan.test_entities.toLocaleString()} test series). Mean ± SD over 5 seeds.`, ['Method', 'MAE', 'WAPE', 'AUROC', 'AUPRC', 'Brier'],
  ['last_value', 'history_mean', 'extratrees', 'supervised', 'ema_jepa', 'sigreg_probe', 'random_probe'].map((m) => [m, ...['mae', 'wape', 'auroc', 'auprc', 'brier'].map((k) => `${f(retail.results[m][k].mean, 4)} ± ${f(retail.results[m][k].seed_sd, 4)}`)]),
  [1600, 1550, 1550, 1550, 1550, 1560]);
TABLE('retailpaired', 'Paired differences (seed-mean predictions), 95% entity-clustered bootstrap CI. Negative ΔMAE favours the first-named method.', ['Comparison', 'ΔMAE', 'ΔBrier'],
  ['ema_jepa_minus_supervised', 'supervised_minus_extratrees', 'sigreg_probe_minus_supervised', 'random_probe_minus_supervised'].map((k) => {
    const v = retail.paired_seed_mean_predictions[k]; return [k.replace(/_minus_/g, ' − '), `${sgn(v.ae.mean, 4)} ${ci(v.ae.ci95, 4)}`, `${sgn(v.brier.mean, 4)} ${ci(v.brier.ci95, 4)}`];
  }), [3800, 2780, 2780]);
FIGURE('retail', 'f13_retail.png', 'FreshRetailNet attribution: sales WAPE (left) and stockout AUROC (right). Purple: identical network under four training regimes.');
Pa('Conclusion for this track: there is **no evidence of a JEPA-specific benefit**. The 0.5% MAE improvement of supervised neural networks over ExtraTrees is statistically clear but small, and applies to *observed* (censored) sales only.');
H1('28. What we have learned so far');
B('**Established technically:** an information-matched, exactly paired counterfactual protocol on a published simulator; reproducible data generation with a coverage gate; a shared rollout interface for all models; 54 passing unit and integration tests.',
  '**Supported empirically:** supervised delta dynamics learn in-distribution one-step action effects better than zero change; ensemble spread ranks prediction error; learned-model MPC beats the native rule on average.',
  '**Inconclusive:** whether model-based planning can beat tuned rules given a larger search budget or longer horizon; whether an uncertainty penalty helps planning.',
  '**Contradicted by current results:** that JEPA-style latent objectives improve dynamics or forecasting here; that the latent architecture is action-sensitive; that learned dynamics transfer to shifted scenarios.',
  '**Untested:** wider action spaces, other topologies, disruptions during iid training, retail-to-simulator transfer, real monetary costs.');

// ---------------- Part VIII
PART('Part VIII — Problems, limitations and open questions');
H1('29. Current technical problems');
TABLE('problems', 'Prioritised problems grounded in this audit.', ['Problem', 'Evidence', 'Why it matters', 'Status', 'Required investigation'], [
  ['Learned dynamics fail out of distribution', 'P1 OOD: no model beats zero-change', 'Planning in new conditions would be misled', 'Open', 'Broader training scenarios; structured/hybrid models'],
  ['Latent models nearly action-blind', 'Sweep response 12–35% of true (iid), 0% (OOD)', 'A planner cannot exploit a model that ignores actions', 'Open', 'Action-centric objectives; inverse-dynamics or action-contrastive terms'],
  ['No planning advantage over tuned rule', 'All learned MPC Δ vs base-stock > 0', 'H4 unsupported', 'Open', 'Search budget/horizon curves; warm-start ablation'],
  ['Restricted action space', 'Only dispatch requests controllable', 'Limits practical relevance', 'Open', 'Procurement control or another environment'],
  ['Single fixed topology, 3 items', '`build_simulation`', 'No transfer claims possible', 'Open', 'Topology variation'],
  ['Legacy timing asymmetry and RNG pairing', 'native saw same-day demand; 1 vs 111 lead-time draws', 'Invalidated legacy comparisons', 'Fixed in v3', '—'],
  ['Legacy coverage gap', '0/288 backlog transitions', 'Explains legacy failure', 'Fixed (v3d gate)', '—'],
  ['Mislabelled legacy "no-action"', '`run_controlled.py` model list', 'Misleading earlier summary', 'Documented', '—'],
  ['Assumed costs', 'Equation 5 weights chosen by project', 'Rankings may change with weights', 'Partial (re-weighting)', 'Re-optimise under alternative weights'],
  ['Uncalibrated uncertainty', 'Spread ranks error, not calibrated', 'No risk guarantee', 'Open', 'Calibration curves, conformal methods'],
  ['ISOMORPH release-to-source mapping', '52,560 vs 7,300-day defaults', 'Provenance of released data', 'Unresolved', 'Authoritative statement from authors'],
  ['Retail sample is a cluster sample', '300 row groups', 'Entity coverage still partial', 'Improved', 'Full-file evaluation'],
  ['No clean-environment install test', 'Only local Python 3.14 run', 'Portability unknown', 'Open', 'Fresh venv / CI'],
], [1800, 2000, 1900, 1100, 2560]);
H1('30. Methodological limitations');
Pa('Simulator results do not prove real-world effectiveness: the network, costs and demand process are synthetic and the action space is narrow. Low training loss did not imply accurate action consequences (latent models had competitive loss but little action sensitivity). Better state prediction did not imply better control (the decoder-only latent model planned well partly because it changed the warm start little). Latent prediction quality did not imply useful representations (`jepa_only`). Forecasting observed retail sales is not recovering demand during stockouts. The legacy runs had 4–8 test episodes, too few for strong conclusions; the v3 runs use 48–60 paired episodes and report CIs, but each model configuration was trained once (planning used seed-0 single models plus one ensemble). Several components changed between the legacy and v3 protocols simultaneously (timing, pairing, data, models), so legacy-to-v3 improvements cannot be attributed to a single change.');
H1('31. Unresolved research questions');
B('Can any objective make a latent world model action-sensitive enough to plan with, on numeric supply-chain states?',
  'Does a structured (graph or hybrid known-mechanics) model generalise to shifted demand and capacity where unstructured models fail?',
  'With a larger search budget or a longer horizon, does model-based planning beat a tuned base-stock rule, or is the decision problem too close to stationary for planning to help?',
  'Does planning exploit model errors (optimiser\'s curse), and does an uncertainty penalty mitigate it?',
  'Which trajectory diversity is necessary: would active, uncertainty-driven data collection improve OOD action effects?',
  'Can representations learned on real retail data transfer to a controllable environment?');

// ---------------- Part IX
PART('Part IX — Reproducibility and codebase guide');
H1('32. How to reproduce');
Pa('Verified environment: macOS (Apple M4, 10 cores), Python 3.14.6, torch 2.12.1, numpy 2.4.6, scikit-learn 1.9.0, pandas 3.0.3; CPU execution. Exact versions: `prototype_new/requirements-verified.txt`. All commands run from the `JEPA/` root and were executed in this round unless marked.');
CODE('PYTHONPATH=. python3 -m pytest prototype_new/tests -q                       # 54 passed',
  'PYTHONPATH=. python3 prototype_new/experiments/collect_decision_data.py --config prototype_new/configs/decision_v3d.json --output prototype_new/results/decision_v3d/data',
  'PYTHONPATH=. python3 prototype_new/experiments/coverage_report.py --data prototype_new/results/decision_v3d/data',
  'PYTHONPATH=. python3 prototype_new/experiments/run_dynamics_v3.py            # ~50 min CPU',
  'PYTHONPATH=. python3 prototype_new/experiments/evaluate_dynamics_v3.py       # P1-P3, ~40 min',
  'PYTHONPATH=. python3 prototype_new/experiments/summarize_dynamics_v3.py',
  'PYTHONPATH=. python3 prototype_new/experiments/run_planning_v3.py --output <dir> --split validation --methods native basestock_k3 ... basestock_k12',
  'PYTHONPATH=. python3 prototype_new/experiments/run_planning_v3.py --output prototype_new/results/decision_v3d/planning_test --split test --methods <methods>',
  'PYTHONPATH=. python3 prototype_new/experiments/summarize_planning_v3.py',
  'PYTHONPATH=. python3 prototype_new/experiments/fetch_retail_subset.py --sample 300 --seed 6101 --output prototype_new/data/fresh_retail_net/data/train_sample300_seed6101.parquet',
  'PYTHONPATH=. python3 prototype_new/experiments/run_retail_attribution.py');
Pa('Generated data, checkpoints and results are written under `prototype_new/results/` and `prototype_new/data/`, which are excluded by `.gitignore`. Runners refuse to overwrite completed result directories. The retail fetch needs network access to Hugging Face; one earlier attempt timed out and a retry mechanism was added. Simulator-reference planning (`mpc_sim_*`) takes roughly 3–5 minutes per episode on this machine.');
H1('33. Codebase navigation guide');
TABLE('code', 'Active components (full inventory in Appendix A).', ['Path', 'Role', 'Status'], [
  ['`src/vendor/isomorph/simulator.py`', 'Upstream simulator (MIT) with dispatch override and split step', 'Active'],
  ['`src/env/isomorph.py`', '`DecisionEnv` (timing, CRN, clone, observation, cost); legacy `IsomorphEnv`', 'Active / legacy class'],
  ['`src/models/controlled.py`', 'Normalizer, persistence, ridge, delta-MLP, latent (4 objectives), ensemble; legacy `ControlledWorldModel`; `sigreg()`', 'Active'],
  ['`src/planning/cem.py`', 'Integer CEM, Goal, receding-horizon controller', 'Active'],
  ['`src/data/protocols.py`, `http_range.py`', 'Split guards, retail validation/windows, bounded HTTP range reader', 'Active'],
  ['`experiments/collect_decision_data.py`, `coverage_report.py`', 'Data generation and coverage gate', 'Active'],
  ['`experiments/run_dynamics_v3.py`, `evaluate_dynamics_v3.py`, `summarize_dynamics_v3.py`', 'Training, paired evaluation, aggregation', 'Active'],
  ['`experiments/run_planning_v3.py`, `summarize_planning_v3.py`', 'Planning comparison', 'Active'],
  ['`experiments/run_retail_attribution.py`, `fetch_retail_subset.py`', 'Real-data track', 'Active'],
  ['`experiments/run_controlled.py`, `evaluate_action_effects.py`, `evaluate_forecast_baseline.py`, `run_isomorph_prediction.py`, `verify_results.py`', 'Legacy controlled path and verifiers', 'Legacy (reproduction)'],
  ['`src/baselines`, `src/evaluation`, `src/losses`, other `src/models/*`, `src/env/{simulator,network,costs,disruptions,actions}.py`', 'First-prototype stack', 'Legacy'],
  ['`tests/`', '14 test modules (54 tests)', 'Active'],
  ['`docs/`', 'Protocols, formulation, reference audit, limitations, this report\'s assets', 'Active'],
], [3900, 3900, 1560]);
H1('34. Software validation findings');
TABLE('tests', 'Validation evidence.', ['Check', 'Result', 'What it establishes'], [
  ['`pytest prototype_new/tests` (pre-edit)', '39 passed', 'Baseline before this round'],
  ['`pytest prototype_new/tests` (final)', '54 passed, 0 failed, 0 skipped', 'Engineering properties incl. new env and model tests'],
  ['`pytest prototype/tests` (legacy tree)', '12 passed', 'Legacy tree unchanged and runnable'],
  ['Upstream parity (2 fixtures)', 'pass', 'Adapter default path equals unmodified simulator'],
  ['Native-via-external equivalence (2 seeds)', 'pass', 'Information-matched native rule is exact'],
  ['Clone independence, CRN pairing, no future information', 'pass', 'Counterfactual branches are valid'],
  ['Replay of 108 test episodes', '0 mismatches', 'Logged data reproduces exactly'],
  ['Legacy summary vs step logs; action-effect probe re-run', 'exact match', 'Legacy numbers reproduce from raw artifacts'],
  ['Retail subset checksum and revision', 'verified at run time', 'Data provenance'],
  ['`compileall`', 'pass', 'Syntax only'],
], [3400, 2200, 3760]);
NOTE('Passing tests validate specific engineering properties (parity, conservation, pairing, shapes). They do not establish that a model has learned useful dynamics or that a planner makes good decisions.');

// ---------------- Part X
PART('Part X — Conclusions and future work');
H1('35. Maturity assessment');
TABLE('maturity', 'Maturity by dimension.', ['Dimension', 'Assessment'], [
  ['Code integrity', 'Good: single shared model interface, preserved legacy code, 54 tests, provenance records.'],
  ['Data integrity', 'Good for the simulator (pinned source, coverage gate, exact replay); partial for retail (cluster sample).'],
  ['Architecture implementation', 'Complete for the stated narrow scope; not graph-structured, no pretrained encoders.'],
  ['Representation learning', 'Implemented; no measured benefit.'],
  ['Action-conditioned prediction', 'Useful in distribution (supervised); weak for latent models; fails OOD.'],
  ['Counterfactual validity', 'Strong protocol (exact pairing); model accuracy limited.'],
  ['Planner functionality', 'Working, feasible, reproducible; budget/horizon not studied.'],
  ['Decision quality', 'Beats native rule; does not beat tuned base-stock.'],
  ['Real-data evidence', 'Competitive observational forecasting; no JEPA benefit; no decision claim.'],
  ['Statistics and reproducibility', 'Episode-clustered CIs, paired designs, pre-registered estimands; single training run per configuration for planning.'],
], [2600, 6760]);
H1('36. Recommended next steps');
B('**Test action-centric objectives** (inverse-dynamics or action-contrastive terms) on the same architecture, with the request-sweep sensitivity ratio as a primary metric. Progress: ratio near 1 with P1 gains preserved.',
  '**Hybrid known-mechanics models** that keep the simulator\'s accounting and learn only demand and lead-time components. The `sim_forecast` reference suggests structure dominates; progress: OOD P1 below zero-change.',
  '**Planner ablations**: remove the native warm start, vary budget and horizon, and plot cost vs compute against base-stock. Progress: a CI excluding zero against the tuned rule.',
  '**Broaden training scenarios and add active data collection** targeted at high-uncertainty states. Progress: OOD P1 and P2 improve.',
  '**Calibrate ensemble uncertainty** and test a fallback to the base-stock rule when spread is high.',
  '**Widen the decision problem** (procurement control or an alternative environment) only after the above, so the action space does not confound model quality.');
FIGURE('roadmap', 'f17_roadmap.png', 'Roadmap. Completed stages have artefacts and tests; planning is partial; robustness, wider actions and real-data transfer are open (proposed direction).', 6.3);
H1('37. Final conclusion');
Pa('The project now has a sound, reproducible experimental foundation: an information-matched controllable simulator, exactly paired counterfactuals, data that covers shortages and recoveries, a common interface for many dynamics models, and pre-registered evaluation. On that foundation, the evidence supports a modest claim: supervised dynamics learn useful in-distribution action effects and support a planner that beats the simulator\'s default rule. It does not support the claims that motivated the project: JEPA-style objectives gave no measurable benefit on simulated dynamics or real retail forecasting, learned action effects did not survive distribution shift, and model-based planning did not beat a simple, tuned rule. The most important open question is whether *any* learned model can capture action consequences robustly enough — beyond what known mechanics plus forecasting already provide — for planning to add value in this setting.');

// ---------------- Appendices
PART('Appendices');
H1('Appendix A — Source inventory');
TABLE('inv', 'prototype_new directory inventory.', ['Path', 'Contents'], [
  ['`src/env/`', 'isomorph.py (active adapter); simulator.py, network.py, costs.py, disruptions.py, actions.py (legacy)'],
  ['`src/vendor/isomorph/`', 'simulator.py, LICENSE (MIT), PROVENANCE.json'],
  ['`src/models/`', 'controlled.py (active); decision_aligned_jepa.py, dynamics.py, jepa_encoder.py, readouts.py, uncertainty.py (legacy); LEWM_LICENSE'],
  ['`src/planning/`', 'cem.py'],
  ['`src/data/`', 'protocols.py, http_range.py'],
  ['`src/baselines/`, `src/evaluation/`, `src/losses/`', 'Legacy benchmark stack'],
  ['`experiments/`', '25 entry points (active v3, legacy controlled, legacy prototype; see Section 33)'],
  ['`configs/`', 'decision_v3, v3b, v3c, v3d; dynamics_v3; planning_v3; retail_attribution; controlled_smoke/development; retail_audited/cache_smoke; legacy configs'],
  ['`tests/`', 'test_decision_env, test_dynamics_v3, test_controlled, test_data_protocols, test_http_range + 9 legacy modules'],
  ['`docs/`', 'PROBLEM_FORMULATION, ARCHITECTURE, DATA_PROTOCOL, EXPERIMENT_PROTOCOL, REFERENCE_AUDIT, LIMITATIONS_AND_CLAIMS, WORKSPACE_INVENTORY, report_assets/'],
  ['`results/` (ignored)', 'decision_v3a–d, retail_attribution, round_claude_audit, controlled_*, isomorph_prediction, retail_*, execution_audit, legacy files'],
  ['Outside prototype_new', 'round_claude/ (plan, baseline report), prototype/ (legacy), vjepa2-main, le-wm-main, reference_SupplyChain_JEPA, SupplyChain_JEPA_outsourced, historical proposal/plan/report files'],
], [3000, 6360]);
H1('Appendix B — Experiment inventory');
TABLE('exp', 'Experiments, seeds and outputs.', ['Experiment', 'Config / seeds', 'Output', 'Status'], [
  ['Legacy smoke planning', 'controlled_smoke.json, seed 1701', 'results/controlled_smoke', 'Reproduced; superseded'],
  ['Legacy development planning', 'controlled_development.json, seed 2701', 'results/controlled_development', 'Reproduced; superseded'],
  ['Legacy action-effect probe', 'frozen member 0', 'results/controlled_development/action_effects.json', 'Reproduced exactly'],
  ['Track A prediction', 'seeds 4701–4720', 'results/isomorph_prediction', 'Existing; diagnostic'],
  ['decision_v3a–c data', 'seeds 31000–, 32000–, 33000–', 'results/decision_v3a..c', 'Failed coverage gate; preserved'],
  ['decision_v3d data', 'seeds 34000–34567', 'results/decision_v3d/data', 'Gate pass'],
  ['v3 dynamics training', 'dynamics_v3.json, model seeds 0–4 / 0–2', 'results/decision_v3d/models', 'Complete'],
  ['v3 evaluation P1–P3', '60 iid + 48 OOD episodes', 'results/decision_v3d/evaluation', 'Complete'],
  ['Base-stock tuning', '60 validation scenarios', 'results/decision_v3d/planning_validation_basestock', 'Complete (k = 9)'],
  ['v3 planning', planning_v3_status(), 'results/decision_v3d/planning_test', plan ? 'Complete' : 'Pending'],
  ['Retail legacy', 'retail_audited.json, seed 5701', 'results/retail_audited_verified', 'Existing'],
  ['Retail attribution', 'retail_attribution.json, seeds 0–4', 'results/retail_attribution', 'Complete'],
], [2400, 2600, 2900, 1460]);
function planning_v3_status() { return plan ? `planning_v3.json, ${plan.episodes} test scenarios` : 'planning_v3.json'; }
H1('Appendix C — Detailed v3 metrics');
TABLE('p3', 'P3 open-loop rollout MAE (units) on iid test, by feature group and horizon.', ['Model', 'Backlog h1', 'Backlog h7', 'Backlog h14', 'Pipeline h1', 'Pipeline h7', 'Pipeline h14', 'Cost7 abs err'],
  ['persistence', 'train_mean_delta', 'ridge', 'delta_mlp', 'delta_mlp_ensemble', 'latent_sup', 'ema', 'sigreg', 'jepa_only', 'sim_forecast'].map((m) => {
    const x = E('test', m); return [m, ...['dest_backlog_h1', 'dest_backlog_h7', 'dest_backlog_h14', 'pipeline_h1', 'pipeline_h7', 'pipeline_h14', 'cost7_abs_err'].map((k) => f(x[k].mean, 1))];
  }), [1700, 1050, 1050, 1050, 1050, 1050, 1050, 1310]);
H1('Appendix D — Commands');
Pa('See Section 32. Report regeneration:');
CODE('PYTHONPATH=. python3 prototype_new/docs/report_assets/make_figures.py',
  'NODE_PATH=<directory containing node_modules/docx> node prototype_new/docs/report_assets/build_report.js');
H1('Appendix E — Known issues and discrepancies');
B('The previous change report\'s "no-action" planner is an action-blind model planner (Section 24.1).',
  'The earlier reference audit described V-JEPA 2-AC as using EMA targets; the local source shows a frozen encoder in the AC stage (corrected in `docs/REFERENCE_AUDIT.md`).',
  'The baseline report path given in the implementation plan (`JEPA/docs/CHANGE_REPORT_2026-10-17_BASELINE.md`) does not exist; the file is at `round_claude/CHANGE_REPORT_2026-10-17_BASELINE.md` and was byte-identical to the live change report at the start of this round.',
  'ISOMORPH released horizon (52,560 days) vs source README default (7,300 days): release-to-source mapping unresolved.',
  'Legacy all-learned-planner identical smoke cost (8,893.06) indicates degenerate planner behaviour in that run.',
  'A normaliser edge case (constant non-zero feature drift masked as no change) was found by a unit test and fixed; no v3d feature was affected (verified).',
  'Planning compared single seed-0 models and one ensemble; training-seed variance of planning cost was not measured.',
  'Rendering of this .docx to PDF could not be performed locally (no LibreOffice/Poppler installed); see the delivery note.');
H1('Appendix F — Glossary');
TABLE('gloss', 'Glossary.', ['Term', 'Meaning'], [
  ['JEPA', 'Joint-Embedding Predictive Architecture: predicts the latent encoding of a target from the encoding of a context.'],
  ['Latent representation', 'Vector produced by an encoder; not directly an observable quantity.'],
  ['Target encoder', 'Encoder producing the prediction target; may be an EMA copy, frozen, or the online encoder.'],
  ['EMA', 'Exponential moving average of parameters: Ē ← mĒ + (1−m)E.'],
  ['SIGReg', 'Regulariser pushing encodings towards an isotropic Gaussian via random projections and characteristic functions (LeWM / LeJEPA).'],
  ['World model', 'Model predicting future states given current state and actions.'],
  ['Action conditioning', 'Making predictions depend on the chosen action.'],
  ['Rollout', 'Repeated application of a model over a sequence of actions.'],
  ['CEM', 'Cross-entropy method: iterative sampling-based optimiser that refits a distribution to the best samples.'],
  ['MPC / receding horizon', 'Plan over a horizon, execute the first action, observe, replan.'],
  ['Counterfactual prediction', 'Predicting what would happen under an action that was not taken, from the same state and randomness.'],
  ['CRN', 'Common random numbers: identical exogenous randomness across compared branches.'],
  ['Backlog', 'Demand owed to customers after a stockout.'],
  ['Stockout', 'Product unavailable; observed sales then understate demand (censoring).'],
  ['Fill rate', 'Share of demand served from stock on the same day.'],
  ['WAPE', 'Weighted absolute percentage error: Σ|error| / Σ actual.'],
  ['AUROC', 'Probability a random positive is ranked above a random negative.'],
  ['Base-stock rule', 'Dispatch up to a target position (here k × demand EMA plus backlog minus on-hand and in-transit).'],
], [2400, 6960]);
H1('References');
Pa('External sources used for background; all project results come from the workspace files cited in the text.');
B('Assran, M., Bardes, A., Fan, D., et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv:2506.09985. Code: github.com/facebookresearch/vjepa2 (MIT).',
  'Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., Balestriero, R. (2026). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv:2603.19312. Code: github.com/lucas-maes/le-wm (MIT).',
  'Assran, M., Duval, Q., Misra, I., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR 2023; arXiv:2301.08243.',
  'LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. OpenReview position paper.',
  'SupplyChain-JEPA contributors. SupplyChain-JEPA: A Predictive World Model for Partially Observed Supply Chains (software, v0.1.0). github.com/htizhang/SupplyChain-JEPA. No open-source licence granted.',
  'ISOMORPH supply-chain simulator, github.com/tuhinsahai/ISOMORPH, commit 9ab2689cbbe02ef5e9bb614f5e9ea60df159abc7 (MIT); dataset huggingface.co/datasets/Isomorph2026/isomorph-supply-chain-benchmark, revision 478ee4d339eeab4a4208b32c6a4b4a2881c999a5 (CC BY 4.0).',
  'Wang, Y., Gu, J., Long, L., Li, X., Shen, L., Fu, Z., Zhou, X., Jiang, X. (2025). FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail. arXiv:2505.16319. Data: huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K, revision c374e85975c82816c495acfa904cc55ab14a47d3 (CC BY 4.0).',
  'Chua, K., Calandra, R., McAllister, R., Levine, S. (2018). Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models. NeurIPS 2018; arXiv:1805.12114.',
  'de Boer, P.-T., Kroese, D. P., Mannor, S., Rubinstein, R. Y. (2005). A Tutorial on the Cross-Entropy Method. Annals of Operations Research 134, 19–67.',
  'Hubbs, C. D., Perez, H. D., Sarwar, O., Sahinidis, N. V., Grossmann, I. E., Wassick, J. M. (2020). OR-Gym: A Reinforcement Learning Library for Operations Research Problems. arXiv:2008.06319.');
H2('Internal evidence map');
TABLE('evid', 'Major claims and their sources.', ['Claim', 'Source', 'Verification'], [
  ['Legacy planning costs', 'results/controlled_development/summary.json, planning_steps.jsonl', 'Recomputed from step logs'],
  ['Legacy action-effect probe', 'results/controlled_development/action_effects.json', 'Re-run from checkpoints, identical'],
  ['Timing and RNG asymmetry', 'src/vendor/isomorph/simulator.py step(); probe in round_claude_audit', 'Code inspection + measurement'],
  ['Coverage gates', 'results/decision_v3*/data/coverage_report.json', 'Generated by coverage_report.py'],
  ['P1–P3, sensitivity, uncertainty', 'results/decision_v3d/evaluation/summary.json, raw.json', 'Generated; 0 replay mismatches'],
  ['Planning', 'results/decision_v3d/planning_test/summary.json, episodes.jsonl', 'Generated; paired seeds'],
  ['Retail attribution', 'results/retail_attribution/metrics.json, data_manifest.json', 'Checksum-verified input'],
  ['Reference identities', 'docs/REFERENCE_AUDIT.md; local LICENSE/README/CITATION files', 'Local file inspection'],
], [2600, 4400, 2360]);

// ================================================================== DOCUMENT
const children = render();
const titlePage = [
  new Paragraph({ spacing: { before: 2600 }, children: [new TextRun({ text: 'JEPA-Based World Models for Supply-Chain Decision-Making', size: 52, bold: true, color: COLORS.head })] }),
  new Paragraph({ spacing: { before: 240, after: 600 }, children: [new TextRun({ text: 'Architecture, Implementation, Experiments, Findings and Open Research Challenges', size: 30, color: COLORS.accent })] }),
  new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: COLORS.accent, space: 1 } }, children: [] }),
  ...[['Report date', today], ['Project directory', 'JEPA/prototype_new'], ['Document status', 'Technical and experimental status report'],
    ['Scope', 'Audited state of the workspace on the report date; supersedes no source files or results']].map(([k, v]) => new Paragraph({ spacing: { before: 160 },
    children: [new TextRun({ text: `${k}: `, bold: true, size: 22 }), new TextRun({ text: v, size: 22 })] })),
  new Paragraph({ spacing: { before: 800 }, children: [new TextRun({ text: 'All numerical results are generated from files in prototype_new/results/ at build time. Costs from the simulator are assumed accounting units, not currency.', italics: true, size: 20, color: '595959' })] }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: 'Contents', size: 32, bold: true, color: COLORS.head })] }),
  new TableOfContents('Contents', { hyperlink: true, headingStyleRange: '1-2' }),
  new Paragraph({ spacing: { before: 200 }, children: [new TextRun({ text: 'If page numbers are missing, right-click the table and choose Update Field.', italics: true, size: 18, color: '808080' })] }),
];

const doc = new Document({
  creator: 'prototype_new report generator', title: 'JEPA-Based World Models for Supply-Chain Decision-Making',
  description: 'Technical and experimental status report', features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 32, bold: true, color: COLORS.head, font: FONT }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 26, bold: true, color: COLORS.accent, font: FONT }, paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 22, bold: true, color: COLORS.head, font: FONT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } }, titlePage: true },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: 'JEPA supply-chain world models — technical and experimental status report', size: 16, color: '808080' })] })] }) },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ['Page ', PageNumber.CURRENT, ' of ', PageNumber.TOTAL_PAGES], size: 16, color: '808080' })] })] }),
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
    },
    children: [...titlePage, ...children],
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log('wrote', OUT, buf.length, 'bytes;', figN, 'figures;', tabN, 'tables'); });
