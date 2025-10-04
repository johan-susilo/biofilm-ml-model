/**
 * Mathematical Calculations
 * 
 * This file handles all the number crunching that happens in the interface:
 * 
 * - Converting between different ratio formats (percentages, decimals, counts)
 * - Normalizing enzyme ratios so they add up to 100%
 * - Calculating estimated concentrations and volumes
 * - Validating that numbers are in the right ranges
 * 
 * The calculations here make sure everything stays mathematically consistent
 * even when users input data in different ways.
 */

/**
 * Normalize config dspb/dnase/prok values to 0..1
 */
export function normalizeConfigRatios(cfg) {
  let d = Number(cfg.dspb ?? 0);
  let n = Number(cfg.dnase ?? 0);
  let p = Number(cfg.prok ?? 0);
  
  // If inputs look like 0..1 values, clamp only (don't convert)
  if (d <= 1 && n <= 1 && p <= 1) {
    return { 
      d: Math.max(0, Math.min(1, d)), 
      n: Math.max(0, Math.min(1, n)), 
      p: Math.max(0, Math.min(1, p)) 
    };
  }
  
  // Otherwise, treat as counts and normalize to proportions
  const s = d + n + p;
  if (!isFinite(s) || s <= 0) return { d: 1/3, n: 1/3, p: 1/3 };
  return { d: d / s, n: n / s, p: p / s };
}

/**
 * Convert proportions to integer counts with proper rounding
 */
export function proportionsToIntegers(obj, total=100) {
  const keys = ['DspB','DNase','ProK'];
  const raw = [obj.DspB || obj.DSpB || 0, obj.DNase || 0, obj.ProK || 0];
  const sum = raw.reduce((a,b)=>a+b,0);
  const ints = {};
  
  if (sum <= 0) {
    const base = Math.floor(total / keys.length);
    keys.forEach((k) => (ints[k] = base));
    let rem = total - base * keys.length;
    let i = 0;
    while (rem > 0) {
      ints[keys[i % keys.length]]++;
      i++;
      rem--;
    }
    return ints;
  }
  
  const scaled = raw.map((v) => (v / sum) * total);
  keys.forEach((k, i) => (ints[k] = Math.round(scaled[i])));
  
  let currentSum = Object.values(ints).reduce((a, b) => a + b, 0);
  
  // adjust rounding differences
  while (currentSum !== total) {
    if (currentSum < total) {
      // add to the largest fractional remainder
      const remainders = keys
        .map((k, i) => ({ k, rem: scaled[i] - Math.floor(scaled[i]) }))
        .sort((a, b) => b.rem - a.rem);
      ints[remainders[0].k]++;
    } else {
      // remove 1 from the largest integer value (>0)
      const sorted = keys
        .map((k) => ({ k, val: ints[k] }))
        .filter((x) => x.val > 0)
        .sort((a, b) => b.val - a.val);
      ints[sorted[0].k]--;
    }
    currentSum = Object.values(ints).reduce((a, b) => a + b, 0);
  }
  
  return ints;
}

/**
 * Compute % removal from control and final OD inputs for a row
 */
export function calculateRemoval(row) {
  const controlOD = parseFloat(document.getElementById('control-od').value);
  const finalOD = parseFloat(row.querySelector('.od600').value);
  const removalCell = row.querySelector('.removal-actual');
  
  if (!isNaN(controlOD) && !isNaN(finalOD) && controlOD > 0) {
    const removal = Math.max(0, ((controlOD - finalOD) / controlOD) * 100);
    removalCell.textContent = `${removal.toFixed(1)}%`;
    removalCell.dataset.actual = removal;
  } else {
    removalCell.textContent = '-';
    delete removalCell.dataset.actual;
  }
}

/**
 * Compute estimated concentration for a row
 */
export function computeEstimatedConcForRow(row) {
  // prefer server-provided total stock concentration from lastOptimalMix, otherwise use a sensible default
  const stock = (window.lastOptimalMix && window.lastOptimalMix.total_stock_concentration_mg_ml) 
    ? parseFloat(window.lastOptimalMix.total_stock_concentration_mg_ml) : 1.0;
  
  const dspb = parseFloat(row.querySelector('.ratio-dspb')?.value) || 0;
  const dnase = parseFloat(row.querySelector('.ratio-dnase')?.value) || 0;
  const prok = parseFloat(row.querySelector('.ratio-prok')?.value) || 0;
  const total = dspb + dnase + prok;
  
  if (total === 0 || stock === 0) return null;
  
  // compute fraction each enzyme contributes (based on integer 'counts' proportions)
  const fracD = dspb / total;
  const fracN = dnase / total;
  const fracP = prok / total;
  const dspbConc = stock * fracD;
  const dnaseConc = stock * fracN;
  const prokConc = stock * fracP;
  
  return {
    totalStockConc: stock,
    dspbConc,
    dnaseConc,
    prokConc,
  };
}

/**
 * Update estimated concentration display for a row
 */
export function updateRowEstimatedConc(row) {
  const conc = computeEstimatedConcForRow(row);
  const el = row.querySelector('.estimated-conc');
  if (!el) return;
  
  if (!conc) { 
    el.textContent = '-'; 
    return; 
  }
  
  el.textContent = `DspB ${conc.dspbConc.toFixed(2)} µL, DNase ${conc.dnaseConc.toFixed(2)} µL, ProK ${conc.prokConc.toFixed(2)} µL`;
}

/**
 * Clamp value between 0 and 1
 */
export function clamp01(x) { 
  x = parseFloat(x); 
  return isFinite(x) ? Math.max(0, Math.min(1, x)) : 0; 
}

/**
 * Copy text to clipboard with fallback support
 */
export function copyToClipboard(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text);
  }
  
  return new Promise((resolve, reject) => {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      ok ? resolve() : reject(new Error('execCommand failed'));
    } catch (e) {
      document.body.removeChild(ta);
      reject(e);
    }
  });
}
