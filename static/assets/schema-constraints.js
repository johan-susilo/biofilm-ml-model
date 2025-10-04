/**
 * Input Validation and Constraints
 * 
 * This file makes sure that all the numbers users enter are valid
 * and within safe ranges for the experiments:
 * 
 * - Checks that pH values are reasonable for biological systems
 * - Ensures temperatures won't damage the enzymes
 * - Validates that reaction times make sense
 * - Prevents accidentally entering impossible values
 * 
 * The validation rules come from the machine learning model's training
 * data, so we know these ranges should give meaningful predictions.
 */

import { AppState, HARD_LIMITS } from './config.js';

/**
 * Clamp and optionally fix the sidebar input ranges based on schema stats
 * provided by the backend. UI currently exposes only pH, Temperature,
 * Total_Volume, and biofilm_age_hours; timing/strategy are fixed defaults.
 */
export function applySchemaConstraints() {
  if (!AppState.SCHEMA_STATS) return;
  
  const map = [
    { id: 'fixed-ph', key: 'pH' },
    { id: 'fixed-temp', key: 'Temperature' },
    { id: 'total-volume', key: 'Total_Volume' },
    { id: 'biofilm-age', key: 'biofilm_age_hours' },
  ];
  
  for (const { id, key } of map) {
    const el = document.getElementById(id);
    const st = AppState.SCHEMA_STATS[key];
    if (el && st) {
      const hard = HARD_LIMITS[key] || {};
      const effMin = Math.max(
        typeof st.min === 'number' ? st.min : -Infinity,
        typeof hard.min === 'number' ? hard.min : -Infinity
      );
      const effMax = Math.min(
        typeof st.max === 'number' ? st.max : Infinity,
        typeof hard.max === 'number' ? hard.max : Infinity
      );
      
      if (isFinite(effMin)) el.min = String(effMin);
      if (isFinite(effMax)) el.max = String(effMax);
      
      const cur = parseFloat(el.value);
      if (!isNaN(cur)) {
        el.value = String(Math.min(
          isFinite(effMax) ? effMax : cur, 
          Math.max(isFinite(effMin) ? effMin : cur, cur)
        ));
      }
      
      // If schema indicates a fixed constant (min == max), force value; also respect hard caps
      if (typeof st.min === 'number' && typeof st.max === 'number' && st.min === st.max) {
        let fixed = st.min;
        if (typeof hard.max === 'number' && fixed > hard.max) fixed = hard.max;
        if (typeof hard.min === 'number' && fixed < hard.min) fixed = hard.min;
        el.value = String(fixed);
        el.disabled = true;
        el.title = 'Fixed by training schema';
      }
    }
  }

  // Enforce fixed defaults for pH/age and default temperature
  const phEl = document.getElementById('fixed-ph');
  if (phEl) { 
    phEl.value = '7.2'; 
    phEl.disabled = true; 
    phEl.title = 'Fixed by protocol'; 
  }
  
  const ageEl = document.getElementById('biofilm-age');
  if (ageEl) { 
    ageEl.value = '24'; 
    ageEl.disabled = true; 
    ageEl.title = 'Fixed by protocol'; 
  }
  
  const tEl = document.getElementById('fixed-temp');
  if (tEl && (!tEl.value || isNaN(parseFloat(tEl.value)))) { 
    tEl.value = '43'; 
  }
}
