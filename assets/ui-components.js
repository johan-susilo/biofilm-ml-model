/**
 * User Interface Components
 * 
 * This file handles all the interactive elements that researchers see and use:
 * - Creating new experiment rows in the table
 * - Managing input sliders and number fields
 * - Handling button clicks and user interactions
 * - Updating the results summary section
 * 
 * The goal is to make the interface intuitive so researchers can focus
 * on their experiments rather than figuring out how to use the software.
 */

import { AppState, DOM_ELEMENTS, INPUT_BASE_CLASSES, HARD_LIMITS, DEFAULT_FIXED } from './config.js';
import { normalizeConfigRatios, calculateRemoval, updateRowEstimatedConc, clamp01, copyToClipboard } from './calculations.js';
import { updateAllVisualizations } from './visualizations.js';
import { applySchemaConstraints } from './schema-constraints.js';
import { getPrediction } from './predictions.js';

/**
 * Create a new experiment row and attach listeners
 */
export function createRow(config) {
  AppState.experimentCount++;
  const row = document.createElement('tr');
  row.className = 'hover:bg-gray-50';
  row.dataset.id = AppState.experimentCount;
  const norm = normalizeConfigRatios(config || {});
  
  // Get reaction time from config, default to 24 hours
  const reactionTime = config?.reactionTime || 24;
  
  row.innerHTML = `
    <td class="p-4 text-gray-500 font-medium">${String(AppState.experimentCount).padStart(2, '0')}</td>
    <td class="p-4"><div class="space-y-2">
        <div class="flex items-center gap-2">
          <label class="text-xs font-bold text-gray-500 w-24">Dispersin B</label>
          <input type="range" class="ratio-dspb-slider ratio-slider w-40" min="0" max="1" step="0.01" value="${norm.d}">
          <input type="number" class="ratio-dspb ${INPUT_BASE_CLASSES} w-20" min="0" max="1" step="0.01" value="${norm.d.toFixed(2)}">
        </div>
        <div class="flex items-center gap-2">
          <label class="text-xs font-bold text-gray-500 w-24">DNase I</label>
          <input type="range" class="ratio-dnase-slider ratio-slider w-40" min="0" max="1" step="0.01" value="${norm.n}">
          <input type="number" class="ratio-dnase ${INPUT_BASE_CLASSES} w-20" min="0" max="1" step="0.01" value="${norm.n.toFixed(2)}">
        </div>
        <div class="flex items-center gap-2">
          <label class="text-xs font-bold text-gray-500 w-24">Proteinase K</label>
          <input type="range" class="ratio-prok-slider ratio-slider w-40" min="0" max="1" step="0.01" value="${norm.p}">
          <input type="number" class="ratio-prok ${INPUT_BASE_CLASSES} w-20" min="0" max="1" step="0.01" value="${norm.p.toFixed(2)}">
        </div>
        <div class="flex items-center justify-between mt-1">
          <div class="text-xs text-gray-500">Estimated vol: <span class="estimated-conc font-medium">-</span></div>
        </div>
    </div></td>
    <td class="p-4"><input type="number" class="reaction-time ${INPUT_BASE_CLASSES} w-32 h-10 text-center font-medium" step="1" min="1" value="${reactionTime}"></td>
    <td class="p-4">
      <div class="relative">
        <input type="number" class="od600 ${INPUT_BASE_CLASSES} w-32 h-10 text-center font-medium" step="0.01" min="0" placeholder="Enter OD600">
       
      </div>
    </td>
    <td class="p-4 font-semibold removal-actual">-</td>
    <td class="p-4 font-medium removal-predicted">-</td>
    <td class="p-4">
      <div class="flex flex-col gap-2">
        <button class="get-prediction-btn px-4 py-2 rounded-lg font-semibold text-xs" title="Click to get ML prediction and lock this experiment">Predict</button>
        <div class="flex gap-2">
          <button class="copy-row-btn px-3 py-1.5 rounded-lg font-semibold text-xs bg-white text-gray-700 border border-gray-300 hover:bg-gray-100">Copy</button>
          <button class="delete-row-btn px-3 py-1.5 rounded-lg font-semibold text-xs bg-red-50 text-red-700 border border-red-200 hover:bg-red-100">Delete</button>
        </div>
      </div>
    </td>
  `;
  
  DOM_ELEMENTS.tableBody.appendChild(row);
  attachRowListeners(row);
  return row;
}

/**
 * Attach input listeners for a given table row
 */
export function attachRowListeners(row) {
  // OD600 input change listener
  row.querySelector('.od600').addEventListener('input', () => {
    calculateRemoval(row);
    updateAllVisualizations();
  });
  
  // Reaction time input listener with validation
  const rtEl = row.querySelector('.reaction-time');
  if (rtEl) {
    rtEl.addEventListener('input', () => {
      const min = HARD_LIMITS.Reaction_Time?.min ?? 1;
      if (parseFloat(rtEl.value) < min) rtEl.value = String(min);
      updateAllVisualizations();
    });
  }
  
  // Prediction button
  row.querySelector('.get-prediction-btn').addEventListener('click', (e) =>
    getPrediction(row, e.currentTarget)
  );
  
  // Copy row button
  const copyBtn = row.querySelector('.copy-row-btn');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const cfg = {
        dspb: parseFloat(row.querySelector('.ratio-dspb')?.value) || 0,
        dnase: parseFloat(row.querySelector('.ratio-dnase')?.value) || 0,
        prok: parseFloat(row.querySelector('.ratio-prok')?.value) || 0,
      };
      createRow(cfg);
      updateAllVisualizations();
    });
  }
  
  // Delete row button
  const delBtn = row.querySelector('.delete-row-btn');
  if (delBtn) {
    delBtn.addEventListener('click', () => {
      row.remove();
      updateAllVisualizations();
    });
  }
  
  // Setup ratio controls
  setupRatioControls(row);
  
  // Initialize slider fill visuals and estimated concentration
  const sldD = row.querySelector('.ratio-dspb-slider');
  const sldN = row.querySelector('.ratio-dnase-slider');
  const sldP = row.querySelector('.ratio-prok-slider');
  setSliderFill(sldD);
  setSliderFill(sldN);
  setSliderFill(sldP);
  updateRowEstimatedConc(row);
}

/**
 * Setup ratio control logic with sum=1 constraint
 */
function setupRatioControls(row) {
  const numD = row.querySelector('.ratio-dspb');
  const numN = row.querySelector('.ratio-dnase');
  const numP = row.querySelector('.ratio-prok');
  const sldD = row.querySelector('.ratio-dspb-slider');
  const sldN = row.querySelector('.ratio-dnase-slider');
  const sldP = row.querySelector('.ratio-prok-slider');

  function getVals() {
    return {
      d: Math.max(0, Math.min(1, parseFloat(numD.value) || 0)),
      n: Math.max(0, Math.min(1, parseFloat(numN.value) || 0)),
      p: Math.max(0, Math.min(1, parseFloat(numP.value) || 0)),
    };
  }
  
  function setDisplay(d, n, p) {
    d = Math.max(0, Math.min(1, d));
    n = Math.max(0, Math.min(1, n));
    p = Math.max(0, Math.min(1, p));
    numD.value = d.toFixed(2);
    numN.value = n.toFixed(2);
    numP.value = p.toFixed(2);
    sldD.value = String(d);
    sldN.value = String(n);
    sldP.value = String(p);
    setSliderFill(sldD);
    setSliderFill(sldN);
    setSliderFill(sldP);
    updateAllVisualizations();
    updateRowEstimatedConc(row);
  }

  // Track last two edited keys to decide which values remain fixed
  const editHistory = [];
  function pushHistory(k) {
    const i = editHistory.indexOf(k);
    if (i !== -1) editHistory.splice(i, 1);
    editHistory.push(k);
    if (editHistory.length > 2) editHistory.shift();
  }

  function enforceAfterEdit(active, val) {
    val = Math.max(0, Math.min(1, val));
    pushHistory(active);
    let { d, n, p } = getVals();
    const keys = ['d','n','p'];
    const fixed = editHistory.slice();
    
    if (fixed.length < 2) {
      // Adjust other two proportionally to fill remainder
      if (active === 'd') {
        const rem = Math.max(0, 1 - val);
        const sum = n + p;
        const nn = sum > 0 ? (n / sum) * rem : rem * 0.5;
        const pp = rem - nn;
        setDisplay(val, nn, pp);
      } else if (active === 'n') {
        const rem = Math.max(0, 1 - val);
        const sum = d + p;
        const dd = sum > 0 ? (d / sum) * rem : rem * 0.5;
        const pp = rem - dd;
        setDisplay(dd, val, pp);
      } else {
        const rem = Math.max(0, 1 - val);
        const sum = d + n;
        const dd = sum > 0 ? (d / sum) * rem : rem * 0.5;
        const nn = rem - dd;
        setDisplay(dd, nn, val);
      }
      return;
    }
    
    // Two locked
    const vals = { d, n, p };
    const varKey = keys.find((k) => !fixed.includes(k));
    const otherFixed = fixed.find((k) => k !== active) || fixed[0];

    if (!fixed.includes(active)) {
      // Case A: last two edited are the other two (keep them fixed), set active = 1 - sum(fixed)
      const fixedSum = (vals[fixed[0]] || 0) + (vals[fixed[1]] || 0);
      const newActive = Math.max(0, 1 - fixedSum);
      vals[active] = newActive;
      setDisplay(vals.d, vals.n, vals.p);
      return;
    }

    // Case B: active is one of the fixed pair: clamp active, adjust only the remaining var
    const maxActive = 1 - (vals[otherFixed] || 0);
    const newActive = Math.max(0, Math.min(val, maxActive));
    vals[active] = newActive;
    const fixedSum = newActive + (vals[otherFixed] || 0);
    const newVar = Math.max(0, 1 - fixedSum);
    vals[varKey] = newVar;
    setDisplay(vals.d, vals.n, vals.p);
  }

  // Slider input events
  sldD.addEventListener('input', () => enforceAfterEdit('d', parseFloat(sldD.value) || 0));
  sldN.addEventListener('input', () => enforceAfterEdit('n', parseFloat(sldN.value) || 0));
  sldP.addEventListener('input', () => enforceAfterEdit('p', parseFloat(sldP.value) || 0));
  
  // Number inputs: live update only own slider + fill; commit on change/blur/Enter
  numD.addEventListener('input', () => { 
    const v = clamp01(numD.value); 
    sldD.value = String(v); 
    setSliderFill(sldD); 
  });
  numN.addEventListener('input', () => { 
    const v = clamp01(numN.value); 
    sldN.value = String(v); 
    setSliderFill(sldN); 
  });
  numP.addEventListener('input', () => { 
    const v = clamp01(numP.value); 
    sldP.value = String(v); 
    setSliderFill(sldP); 
  });
  
  ['change','blur'].forEach(evt => {
    numD.addEventListener(evt, () => enforceAfterEdit('d', clamp01(numD.value)));
    numN.addEventListener(evt, () => enforceAfterEdit('n', clamp01(numN.value)));
    numP.addEventListener(evt, () => enforceAfterEdit('p', clamp01(numP.value)));
  });
  
  [numD,numN,numP].forEach(inp => 
    inp.addEventListener('keydown', (e) => { 
      if(e.key === 'Enter'){ 
        e.preventDefault(); 
        e.currentTarget.blur(); 
      }
    })
  );
}

/**
 * Visually fill the range slider based on its value
 */
export function setSliderFill(sl) {
  if (!sl) return;
  const min = parseFloat(sl.min) || 0;
  const max = parseFloat(sl.max) || 1;
  const val = parseFloat(sl.value) || 0;
  const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
  sl.style.background = `linear-gradient(to right, #A8C899 0%, #A8C899 ${pct}%, #e5e7eb ${pct}%, #e5e7eb 100%)`;
}

/**
 * Update summary metrics in the Results Summary section
 */
export function updateSummaryMetrics() {
  const predictions = Array.from(
    document.querySelectorAll('[data-prediction]')
  ).map((el) => parseFloat(el.dataset.prediction));
  const count = predictions.length;
  document.getElementById('summary-count').textContent = count;
  if (count > 0) {
    document.getElementById('summary-best').textContent = `${Math.max(
      ...predictions
    ).toFixed(1)}%`;
    document.getElementById('summary-mean').textContent = `${(
      predictions.reduce((a, b) => a + b, 0) / count
    ).toFixed(1)}%`;
  } else {
    ['summary-best', 'summary-mean'].forEach(
      (id) => (document.getElementById(id).textContent = '-')
    );
  }
}

/**
 * Update prediction analysis metrics (MAE)
 */
export function updatePredictionAnalysis() {
  let totalError = 0,
    validPairs = 0;
  document.querySelectorAll('#table-body tr').forEach((row) => {
    const actualRaw = row.querySelector('.removal-actual')?.dataset.actual;
    const predictedRaw = row.querySelector('[data-prediction]')?.dataset.prediction;
    const actual = actualRaw !== undefined && actualRaw !== null && actualRaw !== '' ? parseFloat(actualRaw) : NaN;
    const predicted = predictedRaw !== undefined && predictedRaw !== null && predictedRaw !== '' ? parseFloat(predictedRaw) : NaN;
    if (!isNaN(actual) && !isNaN(predicted)) {
      totalError += Math.abs(actual - predicted);
      validPairs++;
    }
  });
  document.getElementById('analysis-mae').textContent =
    validPairs > 0 ? `${(totalError / validPairs).toFixed(2)}%` : '-';
}
