/**
 * Application Configuration
 * 
 * This file contains all the settings and constants that control
 * how the application behaves:
 * 
 * - Where to find the prediction server (API_URL)
 * - Valid ranges for experimental parameters (HARD_LIMITS)
 * - Default values for new experiments
 * - Styling classes for consistent appearance
 * 
 * Having all these settings in one place makes it easy to adjust
 * the application for different lab setups or requirements.
 */

/**
 * Determine the API base URL dynamically.
 * Tries multiple strategies to find the correct backend URL.
 */
export const API_URL = (function(){
  try {
    if (window.API_BASE) return String(window.API_BASE).replace(/\/$/, '');
    const { protocol, hostname, port, origin, pathname } = window.location;
    if (protocol.startsWith('http')) {
      if (port === '8000' || pathname.startsWith('/static') || pathname.startsWith('/ui')) {
        return origin.replace(/\/$/, '');
      }
      if (hostname && hostname !== 'localhost' && hostname !== '127.0.0.1') {
        return `${protocol}//${hostname}:8000`;
      }
    }
    if (hostname === '54.237.111.117') return 'http://54.237.111.117:8000';
  } catch (_) {}
  return 'http://127.0.0.1:8000';
})();

// Backend hard caps derived from PredictionRequest in src/api.py
export const HARD_LIMITS = {
  DspB_ratio: { min: 0 },
  DNase_I_ratio: { min: 0 },
  ProK_ratio: { min: 0 },
  Total_Volume: { min: 10 },
  pH: { min: 6.8, max: 8.2 },
  Temperature: { min: 30, max: 50 },
  Reaction_Time: { min: 1 },
  biofilm_age_hours: { min: 12, max: 96 },
};

// Default fixed timing/strategy (removed from UI)
export const DEFAULT_FIXED = {
  Reaction_Time: 24,
};

// CSS classes for consistent input styling
export const INPUT_BASE_CLASSES = 'w-full bg-gray-100 border-gray-300 rounded-lg shadow-sm p-2 text-sm';

// Initial experiment data
export const INITIAL_DATA = [
  {
    dspb: 0.34,
    dnase: 0.33,
    prok: 0.33,
  },
];

// Global application state
export const AppState = {
  experimentCount: 0,
  SCHEMA_STATS: null,
};

// DOM element references cache
export const DOM_ELEMENTS = {
  tableBody: null,
  addRowBtn: null,
  findOptimalMixBtn: null,
  exportCsvBtn: null,
  suggestExperimentBtn: null,
};

/**
 * Initialize DOM element references
 */
export function initializeDOMReferences() {
  DOM_ELEMENTS.tableBody = document.getElementById('table-body');
  DOM_ELEMENTS.addRowBtn = document.getElementById('add-row-btn');
  DOM_ELEMENTS.findOptimalMixBtn = document.getElementById('find-optimal-mix-btn');
  DOM_ELEMENTS.exportCsvBtn = document.getElementById('export-csv-btn');
  DOM_ELEMENTS.suggestExperimentBtn = document.getElementById('suggest-experiment-btn');
}
