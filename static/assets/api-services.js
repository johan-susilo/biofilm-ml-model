/**
 * API Communication Services
 * 
 * This file handles talking to the prediction server:
 * 
 * - Checks if the server is running and models are loaded
 * - Sends prediction requests with experiment data
 * - Gets optimal mixture recommendations from the optimizer
 * - Validates input data before sending it to the server
 * 
 * Think of this as the messenger between the web interface
 * and the machine learning engine running on the server.
 */

import { API_URL, HARD_LIMITS, AppState } from './config.js';

/**
 * Fetch parameter ranges from backend for input validation.
 * These ranges help the UI validate user inputs before sending requests.
 */
export async function fetchSchemaStats() {
  try {
    const response = await fetch(`${API_URL}/training-schema-stats`);
    if (!response.ok) return;
    
    AppState.SCHEMA_STATS = await response.json();
    return AppState.SCHEMA_STATS;
  } catch (e) {
    // Silently ignore if endpoint not available
    console.log('Schema stats not available:', e.message);
    return null;
  }
}

/**
 * Check API connectivity and model status.
 * Updates the status indicator in the UI.
 */
export async function checkApiStatus() {
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');
  
  try {
    const response = await fetch(`${API_URL}/`);
    if (!response.ok) throw new Error('API response not OK');
    
    const data = await response.json();
    
    // Update status indicators to show API is connected
    statusDot.className = 'status-dot bg-green-500';
    statusText.textContent = `API Connected (${data.model})`;
    statusText.className = 'text-sm text-green-700 font-medium';
    
    return { status: 'connected', data };
    
  } catch (error) {
    // Update status indicators to show API is offline
    statusDot.className = 'status-dot bg-red-500 animate-pulse';
    statusText.textContent = 'API Offline';
    statusText.className = 'text-sm text-red-600 font-medium';
    console.error('API connection failed:', error);
    return { status: 'offline', error };
  }
}

/**
 * Validate outgoing payload against backend-provided schema stats
 */
export function validateAgainstSchemaStats(payload) {
  if (!AppState.SCHEMA_STATS) return { ok: true };
  const violations = [];
  const RATIO_KEYS = new Set(['DspB_ratio', 'DNase_I_ratio', 'ProK_ratio']);
  
  for (const [k, v] of Object.entries(payload)) {
    const st = AppState.SCHEMA_STATS[k];
    if (!st || typeof v !== 'number' || !isFinite(v)) continue;
    
    // Skip validation for ratio fields if schema reports 0..0 (indicates unconstrained or engineered away)
    if (RATIO_KEYS.has(k) && st.min === 0 && st.max === 0) continue;
    
    // Apply backend hard caps in addition to schema
    const hard = HARD_LIMITS[k] || {};
    const effMin = Math.max(
      typeof st.min === 'number' ? st.min : -Infinity,
      typeof hard.min === 'number' ? hard.min : -Infinity
    );
    const effMax = Math.min(
      typeof st.max === 'number' ? st.max : Infinity,
      typeof hard.max === 'number' ? hard.max : Infinity
    );
    
    // If schema is a fixed constant, auto-clamp instead of erroring
    if (typeof st.min === 'number' && typeof st.max === 'number' && st.min === st.max) {
      // treat as informational in UI; don't block if differs
      continue;
    }
    
    if (isFinite(effMin) && v < effMin) violations.push(`${k} < min (${v} < ${effMin})`);
    if (isFinite(effMax) && v > effMax) violations.push(`${k} > max (${v} > ${effMax})`);
  }
  
  if (violations.length) return { ok: false, message: `Out of range: ${violations.join(', ')}` };
  return { ok: true };
}

/**
 * Call /predict API endpoint for a given payload
 */
export async function callPredictAPI(payload) {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  
  const result = await response.json();
  
  if (!response.ok) {
    throw new Error(
      (result.detail && result.detail[0]?.msg) ||
        result.detail ||
        'Prediction failed.'
    );
  }
  
  return result;
}

/**
 * Call /optimal-mix API endpoint
 */
export async function callOptimalMixAPI(payload) {
  const response = await fetch(`${API_URL}/optimal-mix`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  
  const result = await response.json();
  
  if (!response.ok) {
    throw new Error(result.detail || 'Optimization failed');
  }
  
  return result;
}

/**
 * Fetch feature importance data for visualization
 */
export async function fetchFeatureImportance() {
  try {
    const response = await fetch(`${API_URL}/feature-importance`);
    if (!response.ok) throw new Error('Feature importance not available');
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to fetch feature importance:', error);
    return null;
  }
}

/**
 * Load optimal mix from server (used during initialization)
 */
export async function loadOptimalMixFromServer() {
  try {
    const response = await fetch(`${API_URL}/optimal-mix`);
    if (!response.ok) return null;
    
    const result = await response.json();
    return result;
  } catch (e) {
    console.debug('Could not fetch optimal on load', e);
    return null;
  }
}

/**
 * Call /suggest-experiments API endpoint for active learning suggestions
 */
export async function callSuggestExperimentsAPI(payload) {
  const response = await fetch(`${API_URL}/suggest-experiments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  
  const result = await response.json();
  
  if (!response.ok) {
    throw new Error(result.detail || 'Failed to get experiment suggestions');
  }
  
  return result;
}
