/**
 * Machine Learning Predictions
 * 
 * This file handles communication with the AI model to get predictions
 * about biofilm removal effectiveness:
 * 
 * - Sends experiment data to the machine learning model
 * - Gets back predictions about how well the enzymes will work
 * - Finds optimal enzyme combinations using smart optimization
 * - Locks experiment rows after getting predictions (to preserve results)
 * 
 * The machine learning happens on the server, so this file just
 * packages up the data and displays the results in a user-friendly way.
 */

import { DEFAULT_FIXED, DOM_ELEMENTS } from './config.js';
import { callPredictAPI, callOptimalMixAPI, validateAgainstSchemaStats } from './api-services.js';
import { proportionsToIntegers, updateRowEstimatedConc } from './calculations.js';
import { updateAllVisualizations } from './visualizations.js';
import { createRow } from './ui-components.js';

/**
 * Call /predict for a row and lock the row upon success
 */
export async function getPrediction(row, button) {
  button.disabled = true;
  button.textContent = '...';
  const predictionCell = row.querySelector('.removal-predicted');
  predictionCell.innerHTML =
    '<span class="animate-pulse text-gray-400">Calculating...</span>';
    
  const payload = {
    DspB_ratio: parseFloat(row.querySelector('.ratio-dspb').value),
    DNase_I_ratio: parseFloat(row.querySelector('.ratio-dnase').value),
    ProK_ratio: parseFloat(row.querySelector('.ratio-prok').value),
    pH: parseFloat(document.getElementById('fixed-ph').value),
    Temperature: parseFloat(document.getElementById('fixed-temp').value),
    Total_Volume: parseFloat(document.getElementById('total-volume').value),
    biofilm_age_hours: parseFloat(document.getElementById('biofilm-age').value),
    Reaction_Time: parseFloat(row.querySelector('.reaction-time')?.value) || DEFAULT_FIXED.Reaction_Time,
  };
  
  // Normalize integer counts to proportions before validation/sending
  {
    const s = payload.DspB_ratio + payload.DNase_I_ratio + payload.ProK_ratio;
    if (isFinite(s) && s > 0) {
      payload.DspB_ratio = payload.DspB_ratio / s;
      payload.DNase_I_ratio = payload.DNase_I_ratio / s;
      payload.ProK_ratio = payload.ProK_ratio / s;
    }
  }
  
  // Validate against schema stats prior to sending
  const vr = validateAgainstSchemaStats(payload);
  if (!vr.ok) {
    predictionCell.innerHTML = `<span class="text-red-500 has-tooltip relative">Invalid<div class="tooltip w-64">${vr.message}</div></span>`;
    button.disabled = false;
    button.textContent = 'Predict';
    return;
  }
  
  try {
    const result = await callPredictAPI(payload);
    
    predictionCell.innerHTML = `
      <div class="flex items-center gap-3">
          <span class="font-bold text-lg text-gray-800" data-prediction="${result.mean_prediction}">${result.mean_prediction.toFixed(1)}%</span>
          <div class="progress-outer"><div class="progress-inner" style="width: 0%"></div></div>
      </div>
      <div class="text-xs text-gray-500 mt-1 has-tooltip relative">
          Prediction interval: [${result.prediction_interval_low.toFixed(1)}% - ${result.prediction_interval_high.toFixed(1)}%] &middot; Uncertainty: ${result.epistemic_uncertainty.toFixed(1)}%
      </div>`;
      
    const bar = predictionCell.querySelector('.progress-inner');
    if (bar) {
      requestAnimationFrame(() => {
        bar.style.width = `${Math.min(100, result.mean_prediction)}%`;
      });
    }
    
    // Lock this row's controls (ratio inputs + reaction time) so they can't be changed after prediction
    // Keep OD600 editable for actual experimental results
    try {
      row.dataset.locked = '1';
      const lockInputs = [
        row.querySelector('.ratio-dspb'),
        row.querySelector('.ratio-dnase'),
        row.querySelector('.ratio-prok'),
        row.querySelector('.ratio-dspb-slider'),
        row.querySelector('.ratio-dnase-slider'),
        row.querySelector('.ratio-prok-slider'),
        row.querySelector('.reaction-time'), // Also lock reaction time
      ].filter(Boolean);
      
      lockInputs.forEach(input => {
        input.disabled = true;
        input.style.opacity = '0.6';
      });
      
      // Update button to show locked state
      button.disabled = true;
      button.textContent = 'Locked';
      button.classList.add('locked');
      button.title = 'This experiment is locked after prediction. Enzyme ratios and reaction time cannot be changed.';
    
      
      // Auto-remove notification after 3 seconds
      setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
          if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
          }
        }, 300);
      }, 3000);
      
    } catch (_) {
      // Ignore lock errors
    }
    
  } catch (error) {
    predictionCell.innerHTML = `<span class="text-red-500 has-tooltip relative">Error<div class="tooltip w-48">${error.message}</div></span>`;
    button.disabled = false;
    button.textContent = 'Predict';
  } finally {
    updateAllVisualizations();
  }
}

/**
 * Call /optimal-mix and append a recommended row (auto-predict)
 */
export async function findOptimalMix() {
  DOM_ELEMENTS.findOptimalMixBtn.disabled = true;
  DOM_ELEMENTS.findOptimalMixBtn.textContent = 'Optimizing...';

  const firstRow = DOM_ELEMENTS.tableBody.querySelector('tr');
  if (!firstRow) {
    alert('Please add at least one experiment row.');
    DOM_ELEMENTS.findOptimalMixBtn.disabled = false;
    DOM_ELEMENTS.findOptimalMixBtn.textContent = 'Find Optimal Mix';
    return;
  }

  // Gather prior experiments with actual results to guide the optimization
  const prior_experiments = Array.from(DOM_ELEMENTS.tableBody.querySelectorAll('tr'))
    .map((row) => {
      const actualEl = row.querySelector('.removal-actual');
      const actual = actualEl?.dataset.actual ? parseFloat(actualEl.dataset.actual) : null;
      if (actual === null || isNaN(actual)) return null;
      
      return {
        DspB_ratio: parseFloat(row.querySelector('.ratio-dspb')?.value) || 0,
        DNase_I_ratio: parseFloat(row.querySelector('.ratio-dnase')?.value) || 0,
        ProK_ratio: parseFloat(row.querySelector('.ratio-prok')?.value) || 0,
        Reaction_Time: parseFloat(row.querySelector('.reaction-time')?.value) || DEFAULT_FIXED.Reaction_Time,
        biofilm_removal_percentage: actual,
      };
    })
    .filter(Boolean);

  const payload = {
    fixed_conditions: {
      pH: parseFloat(document.getElementById('fixed-ph').value),
      Temperature: parseFloat(document.getElementById('fixed-temp').value),
      Total_Volume: parseFloat(document.getElementById('total-volume').value),
      biofilm_age_hours: parseFloat(document.getElementById('biofilm-age').value),
      Reaction_Time: DEFAULT_FIXED.Reaction_Time,
    },
    prior_experiments: prior_experiments,
  };

  try {
    const result = await callOptimalMixAPI(payload);
    
    // Create new row with optimal ratios
    const intsArr = Array.isArray(result.integer_counts) ? result.integer_counts : null;
    const ratiosArr = Array.isArray(result.ratios) ? result.ratios : null;
    
    let config = { dspb: 0.33, dnase: 0.33, prok: 0.34 }; // fallback
    
    if (intsArr && intsArr.length >= 3) {
      const s = (intsArr[0] || 0) + (intsArr[1] || 0) + (intsArr[2] || 0) || 1;
      config = {
        dspb: (intsArr[0] || 0) / s,
        dnase: (intsArr[1] || 0) / s,
        prok: (intsArr[2] || 0) / s,
      };
    } else if (ratiosArr && ratiosArr.length >= 3) {
      const intsObj = proportionsToIntegers({ 
        DspB: ratiosArr[0] || 0, 
        DNase: ratiosArr[1] || 0, 
        ProK: ratiosArr[2] || 0 
      }, 100);
      const s = (intsObj.DspB || 0) + (intsObj.DNase || 0) + (intsObj.ProK || 0) || 1;
      config = {
        dspb: (intsObj.DspB || 0) / s,
        dnase: (intsObj.DNase || 0) / s,
        prok: (intsObj.ProK || 0) / s,
      };
    }
    
    const newRow = createRow(config);
    
    // Store result for concentration calculations
    const lom = intsArr 
      ? { DspB: intsArr[0] || 0, DNase: intsArr[1] || 0, ProK: intsArr[2] || 0 }
      : proportionsToIntegers({ 
          DspB: ratiosArr?.[0] || 0, 
          DNase: ratiosArr?.[1] || 0, 
          ProK: ratiosArr?.[2] || 0 
        }, 100);
    
    window.lastOptimalMix = { 
      ...lom, 
      total_stock_concentration_mg_ml: result.total_stock_concentration_mg_ml 
    };
    
    updateRowEstimatedConc(newRow);
    
    // Auto-predict for the new row
    setTimeout(() => {
      const predictBtn = newRow.querySelector('.get-prediction-btn');
      if (predictBtn) predictBtn.click();
    }, 100);
    
  } catch (error) {
    alert(`Optimization failed: ${error.message}`);
  } finally {
    DOM_ELEMENTS.findOptimalMixBtn.disabled = false;
    DOM_ELEMENTS.findOptimalMixBtn.textContent = 'Find Optimal Mix';
  }
}
