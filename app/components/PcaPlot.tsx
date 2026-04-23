"use client";

import { useState } from "react";

/**
 * PcaPlot — two-view PCA component.
 *
 * "Observations" tab: scores scatter — one point per (image × prompt),
 *   coloured by prompt so clusters reveal consistent glove behaviour.
 *
 * "Variables" tab: factor map (correlation circle) — arrows show how
 *   each metric (Precision, Recall, FNR, …) loads onto PC1/PC2.
 *   Arrow length = quality of representation in 2-D (cos²).
 *   Variables pointing the same way are positively correlated.
 *
 * PCA via power iteration + deflation — no extra dependencies.
 */

// ── Public input type ────────────────────────────────────────────────────────

export type PcaPoint = {
  prompt: string;   // used for colour grouping in the scores plot
  detections: number;
  meanConf: number;
  precision?: number;
  recall?: number;
  fnr?: number;
  meanIoU?: number;
};

// ── PCA maths ────────────────────────────────────────────────────────────────

function matMulVec(A: number[][], v: number[]) {
  return A.map((row) => row.reduce((s, a, j) => s + a * v[j], 0));
}
function vecNorm(v: number[]) {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}
function vecNormalize(v: number[]) {
  const n = vecNorm(v);
  return n > 1e-10 ? v.map((x) => x / n) : v.map(() => 0);
}
function powerIterate(A: number[][], iters = 120) {
  let v: number[] = Array.from({ length: A.length }, (_, i) => (i % 2 === 0 ? 1 : -1));
  v = vecNormalize(v);
  for (let i = 0; i < iters; i++) v = vecNormalize(matMulVec(A, v));
  const Av = matMulVec(A, v);
  return { vec: v, val: v.reduce((s, vi, i) => s + vi * Av[i], 0) };
}
function deflate(A: number[][], v: number[], val: number) {
  return A.map((row, i) => row.map((a, j) => a - val * v[i] * v[j]));
}
function standardise(M: number[][]) {
  const n = M.length, p = M[0].length;
  const means = Array.from({ length: p }, (_, j) => M.reduce((s, r) => s + r[j], 0) / n);
  const stds  = Array.from({ length: p }, (_, j) => {
    const v = M.reduce((s, r) => s + (r[j] - means[j]) ** 2, 0) / (n > 1 ? n - 1 : 1);
    return Math.sqrt(v) > 1e-10 ? Math.sqrt(v) : 1;
  });
  return M.map((row) => row.map((x, j) => (x - means[j]) / stds[j]));
}

type PcaResult = {
  scores:       Array<{ x: number; y: number; prompt: string }>;
  /** Correlation loadings: loading[j] = [coord_on_pc1, coord_on_pc2]
   *  = eigenvector_j × sqrt(eigenvalue_k).  Values in [-1, 1]. */
  loadings:     Array<{ name: string; l1: number; l2: number; cos2: number }>;
  pc1Pct:       number;
  pc2Pct:       number;
  featureNames: string[];
};

function runPCA(points: PcaPoint[]): PcaResult | null {
  if (points.length < 2) return null;

  const hasGT  = points.some((p) => p.precision !== undefined);
  const hasIoU = points.some((p) => p.meanIoU  !== undefined);

  const featureNames: string[] = ["Det.", "mConf"];
  if (hasGT)        featureNames.push("Prec", "Rec", "FNR");
  if (hasGT && hasIoU) featureNames.push("mIoU");

  const M: number[][] = points.map((p) => {
    const v: number[] = [p.detections, p.meanConf];
    if (hasGT) v.push(p.precision ?? 0, p.recall ?? 0, p.fnr ?? 0);
    if (hasGT && hasIoU) v.push(p.meanIoU ?? 0);
    return v;
  });

  const Z  = standardise(M);
  const n  = Z.length;
  const pf = Z[0].length;

  // Covariance matrix p×p
  const cov: number[][] = Array.from({ length: pf }, (_, i) =>
    Array.from({ length: pf }, (_, j) =>
      Z.reduce((s, row) => s + row[i] * row[j], 0) / (n > 1 ? n - 1 : 1),
    ),
  );
  const totalVar = cov.reduce((s, _, i) => s + cov[i][i], 0) || 1;

  const { vec: ev1, val: val1 } = powerIterate(cov);
  const { vec: ev2, val: val2 } = powerIterate(deflate(cov, ev1, val1));

  const pc1Pct = Math.max(0, Math.round((val1 / totalVar) * 100));
  const pc2Pct = Math.max(0, Math.round((val2 / totalVar) * 100));

  // Scores (projection of observations onto PCs)
  const scores = points.map((pt, idx) => ({
    x:      Z[idx].reduce((s, z, j) => s + z * ev1[j], 0),
    y:      Z[idx].reduce((s, z, j) => s + z * ev2[j], 0),
    prompt: pt.prompt,
  }));

  // Correlation loadings: l_{jk} = ev_{jk} × sqrt(val_k)
  // These equal the Pearson correlation between standardised variable j and PC k.
  const sqVal1 = Math.sqrt(Math.max(0, val1));
  const sqVal2 = Math.sqrt(Math.max(0, val2));
  const loadings = featureNames.map((name, j) => {
    const l1 = ev1[j] * sqVal1;
    const l2 = ev2[j] * sqVal2;
    return { name, l1, l2, cos2: l1 * l1 + l2 * l2 };   // cos² = quality in 2-D
  });

  return { scores, loadings, pc1Pct, pc2Pct, featureNames };
}

// ── Constants ────────────────────────────────────────────────────────────────

const PALETTE = [
  "#06b6d4", "#a78bfa", "#34d399", "#fb923c",
  "#f472b6", "#facc15", "#60a5fa", "#f87171",
  "#4ade80", "#e879f9",
];

const W   = 320;
const H   = 270;
const PAD = { top: 28, right: 16, bottom: 36, left: 36 };
const PW  = W - PAD.left - PAD.right;
const PH  = H - PAD.top  - PAD.bottom;

function trunc(s: string, n: number) {
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

// ── Scores plot ──────────────────────────────────────────────────────────────

function ScoresPlot({
  result,
  points,
}: {
  result: PcaResult;
  points: PcaPoint[];
}) {
  const { scores, pc1Pct, pc2Pct } = result;
  const uniquePrompts = Array.from(new Set(points.map((p) => p.prompt)));
  const promptColor: Record<string, string> = {};
  uniquePrompts.forEach((pr, i) => { promptColor[pr] = PALETTE[i % PALETTE.length]; });

  const xs = scores.map((s) => s.x);
  const ys = scores.map((s) => s.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rX = maxX - minX || 1, rY = maxY - minY || 1;
  const mg = 0.15;
  const toSvg = (x: number, y: number) => ({
    sx: PAD.left + ((x - minX) / rX) * (1 - 2 * mg) * PW + mg * PW,
    sy: PAD.top  + (1 - (y - minY) / rY) * (1 - 2 * mg) * PH + mg * PH,
  });

  const midX = PAD.left + PW / 2;
  const midY = PAD.top  + PH / 2;
  const nPrompts = uniquePrompts.length;
  const nImages  = nPrompts > 0 ? Math.round(points.length / nPrompts) : points.length;

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%"
        style={{ display: "block" }}
        aria-label="PCA scores — one point per image × prompt">
        <rect x={PAD.left} y={PAD.top} width={PW} height={PH} fill="var(--surface2)" rx={4} />
        <line x1={midX} y1={PAD.top} x2={midX} y2={PAD.top + PH}
          stroke="var(--border)" strokeWidth={0.8} strokeDasharray="3 3" />
        <line x1={PAD.left} y1={midY} x2={PAD.left + PW} y2={midY}
          stroke="var(--border)" strokeWidth={0.8} strokeDasharray="3 3" />
        <text x={PAD.left + PW / 2} y={H - 4} textAnchor="middle" fill="var(--text-dim)" fontSize={9}>
          PC1 ({pc1Pct}% var)
        </text>
        <text x={10} y={PAD.top + PH / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={9}
          transform={`rotate(-90, 10, ${PAD.top + PH / 2})`}>
          PC2 ({pc2Pct}% var)
        </text>
        <text x={PAD.left + PW / 2} y={14} textAnchor="middle" fill="var(--text-dim)"
          fontSize={9} fontWeight={700} letterSpacing={0.5}>
          {points.length} obs — {nImages} img × {nPrompts} prompt{nPrompts !== 1 ? "s" : ""}
        </text>
        {scores.map((s, i) => {
          const { sx, sy } = toSvg(s.x, s.y);
          const color = promptColor[s.prompt];
          return (
            <circle key={i} cx={sx} cy={sy} r={4.5}
              fill={color} fillOpacity={0.8} stroke={color} strokeWidth={1} strokeOpacity={0.4}>
              <title>{s.prompt}</title>
            </circle>
          );
        })}
      </svg>

      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 10px", marginTop: 4 }}>
        {uniquePrompts.map((pr) => (
          <div key={pr} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: "0.62rem" }}>
            <span style={{ width: 9, height: 9, borderRadius: "50%", background: promptColor[pr], flexShrink: 0, display: "inline-block" }} />
            <span style={{ color: "var(--text-dim)" }} title={pr}>{trunc(pr, 32)}</span>
          </div>
        ))}
      </div>
    </>
  );
}

// ── Variables factor map ─────────────────────────────────────────────────────

/** Interpolate between red (#ef4444) and grey based on cos² quality. */
function qualityColor(cos2: number): string {
  // cos2 in [0,1]: 1 = fully represented (bright red), 0 = not (muted)
  const r = Math.round(80  + (239 - 80)  * cos2);
  const g = Math.round(100 + (68  - 100) * cos2);
  const b = Math.round(120 + (68  - 120) * cos2);
  return `rgb(${r},${g},${b})`;
}

function VariablesPlot({ result }: { result: PcaResult }) {
  const { loadings, pc1Pct, pc2Pct } = result;

  // The factor map lives on the unit circle; centre = (cx, cy) in SVG coords.
  const cx = PAD.left + PW / 2;
  const cy = PAD.top  + PH / 2;
  const R  = Math.min(PW, PH) / 2 - 6;   // radius in SVG px

  // loading (l1, l2) are already in [-1, 1] — map to SVG:
  //   x_svg = cx + l1 * R,  y_svg = cy - l2 * R  (flip Y because SVG Y grows down)
  const toSvg = (l1: number, l2: number) => ({
    sx: cx + l1 * R,
    sy: cy - l2 * R,
  });

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%"
        style={{ display: "block" }}
        aria-label="PCA variables factor map">

        <rect x={PAD.left} y={PAD.top} width={PW} height={PH} fill="var(--surface2)" rx={4} />

        {/* Correlation circle */}
        <circle cx={cx} cy={cy} r={R}
          fill="none" stroke="var(--border)" strokeWidth={0.8} strokeDasharray="4 3" />

        {/* Axes */}
        <line x1={cx - R} y1={cy} x2={cx + R} y2={cy}
          stroke="var(--border)" strokeWidth={0.8} strokeDasharray="3 3" />
        <line x1={cx} y1={cy - R} x2={cx} y2={cy + R}
          stroke="var(--border)" strokeWidth={0.8} strokeDasharray="3 3" />

        {/* Axis labels */}
        <text x={cx} y={H - 4} textAnchor="middle" fill="var(--text-dim)" fontSize={9}>
          PC1 ({pc1Pct}% var)
        </text>
        <text x={10} y={cy} textAnchor="middle" fill="var(--text-dim)" fontSize={9}
          transform={`rotate(-90, 10, ${cy})`}>
          PC2 ({pc2Pct}% var)
        </text>

        {/* Title */}
        <text x={cx} y={14} textAnchor="middle" fill="var(--text-dim)"
          fontSize={9} fontWeight={700} letterSpacing={0.5}>
          Variables factor map
        </text>

        {/* Arrows + labels */}
        {loadings.map(({ name, l1, l2, cos2 }) => {
          const { sx, sy } = toSvg(l1, l2);
          const color = qualityColor(cos2);
          // arrowhead offset: back-track 6 px along the arrow direction
          const len = Math.sqrt((sx - cx) ** 2 + (sy - cy) ** 2) || 1;
          const headX = sx - ((sx - cx) / len) * 6;
          const headY = sy - ((sy - cy) / len) * 6;
          // label offset: push 10 px past the tip
          const lx = cx + (sx - cx) * 1.18;
          const ly = cy + (sy - cy) * 1.18;
          const anchor = l1 >= 0 ? "start" : "end";
          return (
            <g key={name}>
              {/* Arrow shaft */}
              <line x1={cx} y1={cy} x2={headX} y2={headY}
                stroke={color} strokeWidth={1.5} />
              {/* Arrowhead (small triangle) */}
              <polygon
                points={`${sx},${sy} ${headX + (sy - cy) / len * 3},${headY - (sx - cx) / len * 3} ${headX - (sy - cy) / len * 3},${headY + (sx - cx) / len * 3}`}
                fill={color}
              />
              {/* Label */}
              <text x={lx} y={ly + 3.5} textAnchor={anchor}
                fill={color} fontSize={9} fontWeight={600}>
                {name}
              </text>
            </g>
          );
        })}

        {/* cos² legend bar (bottom-right) */}
        {(() => {
          const bx = PAD.left + PW - 55, by = PAD.top + PH - 12, bw = 50, bh = 5;
          return (
            <g>
              <defs>
                <linearGradient id="cos2grad" x1="0" x2="1" y1="0" y2="0">
                  <stop offset="0%"   stopColor={qualityColor(0)} />
                  <stop offset="100%" stopColor={qualityColor(1)} />
                </linearGradient>
              </defs>
              <rect x={bx} y={by} width={bw} height={bh} rx={2} fill="url(#cos2grad)" />
              <text x={bx}      y={by - 2} fill="var(--text-muted)" fontSize={7}>low cos²</text>
              <text x={bx + bw} y={by - 2} fill="var(--text-muted)" fontSize={7} textAnchor="end">high cos²</text>
            </g>
          );
        })()}
      </svg>

      {/* cos² table */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "3px 10px", marginTop: 4 }}>
        {loadings.map(({ name, cos2 }) => (
          <div key={name} style={{ display: "flex", alignItems: "center", gap: 3, fontSize: "0.6rem" }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: qualityColor(cos2), display: "inline-block", flexShrink: 0 }} />
            <span style={{ color: "var(--text-dim)" }}>{name}</span>
            <span style={{ color: qualityColor(cos2), fontWeight: 700 }}>{(cos2 * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
      <p style={{ fontSize: "0.6rem", color: "var(--text-muted)", marginTop: 3 }}>
        cos² = % of variable variance captured in this 2-D projection. Arrow colour encodes quality.
      </p>
    </>
  );
}

// ── Root component ───────────────────────────────────────────────────────────

export default function PcaPlot({ points }: { points: PcaPoint[] }) {
  const [view, setView] = useState<"obs" | "var">("obs");
  const result = runPCA(points);

  if (!result || points.length < 2) {
    return (
      <p style={{ fontSize: "0.68rem", color: "var(--text-muted)", padding: "8px 0" }}>
        Need at least 2 data points for PCA (run the batch with 2+ images or prompts).
      </p>
    );
  }

  const uniquePrompts = Array.from(new Set(points.map((p) => p.prompt)));

  return (
    <div>
      {/* Tab strip */}
      <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
        {(["obs", "var"] as const).map((v) => (
          <button
            key={v}
            onClick={() => setView(v)}
            style={{
              fontSize: "0.65rem",
              padding: "2px 10px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              cursor: "pointer",
              background: view === v ? "var(--accent)" : "var(--surface3)",
              color:      view === v ? "#fff"          : "var(--text-dim)",
              fontWeight: view === v ? 700              : 400,
            }}
          >
            {v === "obs" ? "Observations" : "Variables"}
          </button>
        ))}
      </div>

      {view === "obs"
        ? <ScoresPlot result={result} points={points} />
        : <VariablesPlot result={result} />
      }

      {points.length < 6 && (
        <p style={{ fontSize: "0.6rem", color: "var(--text-muted)", marginTop: 4 }}>
          ⚠ Only {points.length} data point{points.length !== 1 ? "s" : ""} ({uniquePrompts.length} prompt{uniquePrompts.length !== 1 ? "s" : ""}) — more images improve PCA reliability.
        </p>
      )}
    </div>
  );
}
