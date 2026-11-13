// Forecast entry for polymarket-nvidia-ceo-forecast-01.
//
// Model (per market, independent): the median forecast at every horizon is
// the last observed price (pure persistence). The forecast distribution of
// the price change is a two-part mixture: with probability p0(h) the price
// is exactly unchanged (a point mass at zero change, shrinking with horizon
// as (1 - lambda)^h), and otherwise it follows a normal distribution whose
// scale grows as h^0.32. The 5% and 95% quantiles get an extra 1.40 width
// factor; the 25% and 75% quantiles do not. See SPEC.md for the full
// behavioral description.
//
// Standard library only, no network, no randomness, fully deterministic.

import * as fs from "node:fs";
import * as path from "node:path";

// ---------------------------------------------------------------------------
// Fixed model constants
// ---------------------------------------------------------------------------

// Minutes of trailing history used to estimate the move scale and rate.
const WINDOW_MINUTES = 120;
// Horizon exponent: forecast width grows as h^ALPHA, not sqrt(h).
const HORIZON_EXPONENT = 0.32;
// Bounds on the estimated one-minute move scale.
const SIGMA_FLOOR = 0.0008;
const SIGMA_CAP = 0.01;
// Extra width on the outer (5% / 95%) quantiles; inner quantiles unscaled.
const OUTER_MULTIPLIER = 1.4;
const INNER_MULTIPLIER = 1.0;
// Prior per-minute probability that the price changes, and its strength.
const MOVE_RATE_PRIOR = 0.13;
const MOVE_RATE_PSEUDO_COUNT = 2;
// Largest allowed probability of "price exactly unchanged after h minutes".
const POINT_MASS_CAP = 0.95;
// Forecast horizon in minutes.
const HORIZON = 240;
// Quantile levels, in output order.
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];

// ---------------------------------------------------------------------------
// Input shapes
// ---------------------------------------------------------------------------

interface MarketInput {
  id: string;
  name: string;
  slug: string;
  yes_token: string;
  start: string;
  prices: number[];
}

interface CaseInput {
  schema: string;
  competition: string;
  case: string;
  at: string;
  horizon_minutes: number;
  quantiles: number[];
  event: unknown;
  markets: MarketInput[];
}

// ---------------------------------------------------------------------------
// Standard normal quantile function (Acklam's rational approximation).
// Accurate to ~1.15e-9 absolute error, far beyond the 1e-7 the model needs.
// ---------------------------------------------------------------------------

const ACKLAM_A = [
  -39.69683028665376, 220.9460984245205, -275.9285104469687,
  138.357751867269, -30.66479806614716, 2.506628277459239,
];
const ACKLAM_B = [
  -54.47609879822406, 161.5858368580409, -155.6989798598866,
  66.80131188771972, -13.28068155288572,
];
const ACKLAM_C = [
  -0.007784894002430293, -0.3223964580411365, -2.400758277161838,
  -2.549732539343734, 4.374664141464968, 2.938163982698783,
];
const ACKLAM_D = [
  0.007784695709041462, 0.3224671290700398, 2.445134137142996,
  3.754408661907416,
];
const ACKLAM_P_LOW = 0.02425;
const ACKLAM_P_HIGH = 1 - ACKLAM_P_LOW;

// Evaluate a polynomial in Horner form: coeffs[0]*t^n + ... + coeffs[n].
function horner(coeffs: number[], t: number): number {
  let acc = coeffs[0];
  for (let i = 1; i < coeffs.length; i++) {
    acc = acc * t + coeffs[i];
  }
  return acc;
}

// Inverse of the standard normal CDF.
function inverseNormal(p: number): number {
  // Safety net; the mixture formulas only ever pass values strictly
  // inside (0, 1), so this clamp should never actually engage.
  const clamped = Math.min(1 - 1e-12, Math.max(1e-12, p));
  if (clamped < ACKLAM_P_LOW) {
    // Lower tail: result is negative.
    const q = Math.sqrt(-2 * Math.log(clamped));
    const num = horner(ACKLAM_C, q);
    const den = horner([...ACKLAM_D, 1], q);
    return num / den;
  }
  if (clamped > ACKLAM_P_HIGH) {
    // Upper tail: mirror of the lower tail.
    const q = Math.sqrt(-2 * Math.log(1 - clamped));
    const num = horner(ACKLAM_C, q);
    const den = horner([...ACKLAM_D, 1], q);
    return -num / den;
  }
  // Central region.
  const q = clamped - 0.5;
  const r = q * q;
  const num = horner(ACKLAM_A, r);
  const den = horner([...ACKLAM_B, 1], r);
  return (q * num) / den;
}

// ---------------------------------------------------------------------------
// Per-market estimation from the trailing WINDOW_MINUTES of prices.
// ---------------------------------------------------------------------------

interface MarketParams {
  // Last observed price (the median forecast at every horizon).
  lastPrice: number;
  // One-minute move scale, floored and capped.
  sigma: number;
  // Per-minute probability the price moves, shrunk toward the prior.
  lambda: number;
}

function estimateParams(prices: number[]): MarketParams {
  const n = prices.length;
  const lastPrice = prices[n - 1];
  // Number of trailing one-minute differences available (at most 120).
  const m = Math.min(WINDOW_MINUTES, n - 1);
  if (m <= 0) {
    // Fewer than two prices: fall back to the floor scale and the prior.
    return { lastPrice, sigma: SIGMA_FLOOR, lambda: MOVE_RATE_PRIOR };
  }
  let sumSquares = 0;
  let moves = 0;
  for (let i = 0; i < m; i++) {
    const diff = prices[n - 1 - i] - prices[n - 2 - i];
    sumSquares += diff * diff;
    if (diff !== 0) moves++;
  }
  const sigmaRaw = Math.sqrt(sumSquares / m);
  const sigma = Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, sigmaRaw));
  const lambda =
    (moves + MOVE_RATE_PSEUDO_COUNT * MOVE_RATE_PRIOR) /
    (m + MOVE_RATE_PSEUDO_COUNT);
  return { lastPrice, sigma, lambda };
}

// ---------------------------------------------------------------------------
// Forecast quantiles for one market at one horizon.
// ---------------------------------------------------------------------------

// Width multiplier for a quantile level: the outer levels get extra width.
function levelMultiplier(tau: number): number {
  if (tau === 0.05 || tau === 0.95) return OUTER_MULTIPLIER;
  return INNER_MULTIPLIER;
}

// Raw (unrounded, unclamped) forecast quantiles at horizon h minutes ahead.
function forecastRowRaw(
  params: MarketParams,
  h: number,
): number[] {
  const { lastPrice: x, sigma, lambda } = params;
  // Horizon scale grows as h^0.32.
  const sigmaH = sigma * Math.pow(h, HORIZON_EXPONENT);
  // Probability the price is exactly unchanged after h minutes.
  const p0 = Math.min(POINT_MASS_CAP, Math.pow(1 - lambda, h));
  // Continuous-part scale, inflated so the mixture variance stays sigmaH^2.
  const s = sigmaH / Math.sqrt(1 - p0);
  // The point mass occupies the probability interval [lo, hi].
  const lo = (1 - p0) / 2;
  const hi = (1 + p0) / 2;
  const continuous = 1 - p0;

  const row: number[] = [];
  for (const tau of LEVELS) {
    let q: number;
    if (tau === 0.5) {
      // Median: always exactly the last price (pure persistence).
      q = x;
    } else if (tau < lo) {
      // Lower continuous tail.
      const z = inverseNormal(tau / continuous);
      q = x + s * z * levelMultiplier(tau);
    } else if (tau > hi) {
      // Upper continuous tail.
      const z = inverseNormal((tau - p0) / continuous);
      q = x + s * z * levelMultiplier(tau);
    } else {
      // Inside the point mass: exactly the last price.
      q = x;
    }
    row.push(q);
  }
  return row;
}

// Finalize one row: clamp to [0, 1], round to 6 decimals, then enforce
// non-decreasing order left to right (in exactly that order).
function finalizeRow(raw: number[]): number[] {
  const out: number[] = [];
  let prev = -1;
  for (const q of raw) {
    let v = Math.min(1, Math.max(0, q));
    v = Math.round(v * 1_000_000) / 1_000_000;
    if (v === 0) v = 0; // never emit -0
    v = Math.max(prev, v);
    out.push(v);
    prev = v;
  }
  return out;
}

// Full 240-row forecast for one market.
function forecastMarket(prices: number[]): number[][] {
  if (prices.length === 0) {
    // Defensive fallback (should not occur in a valid case): flat 0.5.
    const flat: number[][] = [];
    for (let h = 0; h < HORIZON; h++) {
      flat.push([0.5, 0.5, 0.5, 0.5, 0.5]);
    }
    return flat;
  }
  const params = estimateParams(prices);
  const rows: number[][] = [];
  for (let h = 1; h <= HORIZON; h++) {
    rows.push(finalizeRow(forecastRowRaw(params, h)));
  }
  return rows;
}

// ---------------------------------------------------------------------------
// I/O: read every .in file, write one .out file per case.
// ---------------------------------------------------------------------------

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
  fs.mkdirSync(outputDir, { recursive: true });

  const files = fs
    .readdirSync(inputDir)
    .filter((name) => name.endsWith(".in"))
    .sort();

  for (const file of files) {
    const caseId = file.slice(0, -".in".length);
    const raw = fs.readFileSync(path.join(inputDir, file), "utf-8");
    const parsed = JSON.parse(raw) as CaseInput;

    if (
      parsed.quantiles.length !== LEVELS.length ||
      !parsed.quantiles.every((q, i) => q === LEVELS[i])
    ) {
      throw new Error(
        `case ${caseId}: quantiles must be exactly [0.05, 0.25, 0.5, 0.75, 0.95]`,
      );
    }

    const forecasts = parsed.markets.map((market) => ({
      id: market.id,
      q: forecastMarket(market.prices),
    }));

    const outPath = path.join(outputDir, `${caseId}.out`);
    fs.writeFileSync(outPath, JSON.stringify({ forecasts }) + "\n", "utf-8");
  }
}

main();
