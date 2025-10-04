/**
 * Biofilm Prediction Interface - Main Application
 * 
 * This is the main entry point for the biofilm prediction web interface.
 * It connects all the different components together and handles the overall
 * application flow.
 * 
 * What this interface does:
 * - Lets researchers input enzyme ratios and reaction conditions
 * - Gets machine learning predictions for biofilm removal effectiveness
 * - Suggests optimal enzyme combinations using Bayesian optimization
 * - Shows interactive visualizations of the data and results
 * - Exports experiment data to CSV files for further analysis
 * 
 * The interface is built with modern web technologies but kept simple
 * so researchers can focus on their experiments rather than learning
 * complex software.
 */

import { 
  initializeDOMReferences, 
  DOM_ELEMENTS, 
  INITIAL_DATA 
} from './config.js';

import { 
  checkApiStatus, 
  fetchSchemaStats,
  loadOptimalMixFromServer,
  callSuggestExperimentsAPI
} from './api-services.js';

import { 
  rowsToCsv, 
  downloadCsv 
} from './data-export.js';

import { 
  proportionsToIntegers,
  updateRowEstimatedConc 
} from './calculations.js';

import { 
  renderCoreFeatureImportancePlot, 
  updateAllVisualizations 
} from './visualizations.js';

import { 
  createRow 
} from './ui-components.js';

import { 
  applySchemaConstraints 
} from './schema-constraints.js';

import { 
  findOptimalMix 
} from './predictions.js';

/**
 * Initialize the application
 */
async function initializeApp() {
  // Initialize DOM references
  initializeDOMReferences();
  
  // Create initial experiment rows
  INITIAL_DATA.forEach((c) =>
    createRow({
      ...c,
      dspb: c.dspb || 0,
      dnase: c.dnase || 0,
      prok: c.prok || 0,
    })
  );
  
  // Try to apply server-provided optimal mix to first row on startup
  await applyServerOptimalOnLoad();
  
  // Check API status and load data
  const apiStatus = await checkApiStatus();
  if (apiStatus.status === 'connected') {
    await renderCoreFeatureImportancePlot();
    await fetchSchemaStats();
    applySchemaConstraints();
  }
  
  // Setup event listeners
  setupEventListeners();
  
  // Initial visualization update
  updateAllVisualizations();
  
  // Start periodic API status checks
  setInterval(checkApiStatus, 5000);
}

/**
 * Apply server optimal mix to first row during initialization
 */
async function applyServerOptimalOnLoad() {
  try {
    const opt = await loadOptimalMixFromServer();
    if (!opt) return;
    
    const first = DOM_ELEMENTS.tableBody.querySelector('tr');
    if (!first) return;
    
    const intsArr = Array.isArray(opt.integer_counts) ? opt.integer_counts : null;
    const ratiosArr = Array.isArray(opt.ratios) ? opt.ratios : null;
    
    if (intsArr || ratiosArr) {
      const dspbEl = first.querySelector('.ratio-dspb');
      const dnaseEl = first.querySelector('.ratio-dnase');
      const prokEl = first.querySelector('.ratio-prok');
      
      if (dspbEl && dnaseEl && prokEl) {
        if (intsArr) {
          const s = (intsArr[0]||0) + (intsArr[1]||0) + (intsArr[2]||0) || 1;
          dspbEl.value = String(((intsArr[0]||0)/s).toFixed(2));
          dnaseEl.value = String(((intsArr[1]||0)/s).toFixed(2));
          prokEl.value = String(((intsArr[2]||0)/s).toFixed(2));
          // trigger normalization path to sync sliders
          dspbEl.dispatchEvent(new Event('change'));
        } else if (ratiosArr) {
          const intsObj = proportionsToIntegers({ 
            DspB: ratiosArr[0] || 0, 
            DNase: ratiosArr[1] || 0, 
            ProK: ratiosArr[2] || 0 
          }, 100);
          const s2 = (intsObj.DSpB||intsObj.DspB||0) + (intsObj.DNase||0) + (intsObj.ProK||0) || 1;
          const d2 = (intsObj.DspB ?? intsObj.DSpB ?? 0) / s2;
          const n2 = (intsObj.DNase ?? 0) / s2;
          const p2 = (intsObj.ProK ?? 0) / s2;
          dspbEl.value = String(d2.toFixed(2));
          dnaseEl.value = String(n2.toFixed(2));
          prokEl.value = String(p2.toFixed(2));
          dspbEl.dispatchEvent(new Event('change'));
        }
        
        updateRowEstimatedConc(first);
        updateAllVisualizations();
        
        const lom = intsArr
          ? { DspB: intsArr[0] || 0, DNase: intsArr[1] || 0, ProK: intsArr[2] || 0 }
          : proportionsToIntegers({ 
              DspB: ratiosArr?.[0] || 0, 
              DNase: ratiosArr?.[1] || 0, 
              ProK: ratiosArr?.[2] || 0 
            }, 100);
            
        window.lastOptimalMix = { 
          ...lom, 
          total_stock_concentration_mg_ml: opt.total_stock_concentration_mg_ml 
        };
        // Do not auto-predict on load to keep first row unlocked
      }
    }
  } catch (e) {
    console.debug('Could not fetch optimal on load', e);
  }
}

/**
 * Setup event listeners for main application controls
 */
function setupEventListeners() {
  // Add row button
  DOM_ELEMENTS.addRowBtn.addEventListener('click', () =>
    createRow({
      dspb: 0.33,
      dnase: 0.33,
      prok: 0.34,
    })
  );
  
  // Find optimal mix button
  DOM_ELEMENTS.findOptimalMixBtn.addEventListener('click', findOptimalMix);
  
  // Export CSV button
  if (DOM_ELEMENTS.exportCsvBtn) {
    DOM_ELEMENTS.exportCsvBtn.addEventListener('click', () => 
      downloadCsv(rowsToCsv())
    );
  }
  
  // Suggest experiment button
  if (DOM_ELEMENTS.suggestExperimentBtn) {
    DOM_ELEMENTS.suggestExperimentBtn.addEventListener('click', suggestNewExperiments);
  }
}

/**
 * Handle experiment suggestion requests
 * Uses active learning to suggest the most informative experiments
 */
async function suggestNewExperiments() {
  try {
    // Show loading state
    DOM_ELEMENTS.suggestExperimentBtn.disabled = true;
    DOM_ELEMENTS.suggestExperimentBtn.textContent = 'Finding suggestions...';
    
    // Call the API to get experiment suggestions
    const response = await callSuggestExperimentsAPI({ 
      fixed_conditions: {},
      prior_experiments: null
    });
    
    if (response && response.suggestions && response.suggestions.length > 0) {
      displayExperimentSuggestions(response.suggestions);
    } else {
      alert('No experiment suggestions available at this time.');
    }
    
  } catch (error) {
    console.error('Error getting experiment suggestions:', error);
    alert('Failed to get experiment suggestions. Please try again.');
  } finally {
    // Reset button state
    DOM_ELEMENTS.suggestExperimentBtn.disabled = false;
    DOM_ELEMENTS.suggestExperimentBtn.textContent = 'Suggest New Experiment';
  }
}

/**
 * Display experiment suggestions in the suggestions table
 */
function displayExperimentSuggestions(suggestions) {
  const suggestionsSection = document.getElementById('suggestions-section');
  const suggestionsTableBody = document.getElementById('suggestions-table-body');
  
  // Clear previous suggestions
  suggestionsTableBody.innerHTML = '';
  
  // Add each suggestion to the table
  suggestions.forEach((suggestion, index) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td class="px-3 py-2">${index + 1}</td>
      <td class="px-3 py-2">${(suggestion.dspb * 100).toFixed(1)}%</td>
      <td class="px-3 py-2">${(suggestion.dnase * 100).toFixed(1)}%</td>
      <td class="px-3 py-2">${(suggestion.prok * 100).toFixed(1)}%</td>
      <td class="px-3 py-2">${suggestion.reaction_time.toFixed(1)}h</td>
      <td class="px-3 py-2">${suggestion.predicted.toFixed(2)}%</td>
      <td class="px-3 py-2">${suggestion.uncertainty.toFixed(3)}%</td>
      <td class="px-3 py-2">
        <button class="bg-[#A8C899] hover:bg-[#99bb8a] text-[#1A472A] px-4 py-2 rounded-lg text-sm font-semibold shadow-md transition-colors" onclick="useSuggestion(${index})">
          Use This
        </button>
      </td>
    `;
    suggestionsTableBody.appendChild(row);
  });
  
  // Show the suggestions section
  suggestionsSection.style.display = 'block';
  
  // Store suggestions globally for the "Use This" buttons
  window.currentSuggestions = suggestions;
  
  // Scroll to suggestions
  suggestionsSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Use a suggested experiment by adding it as a new row
 */
window.useSuggestion = function(index) {
  const suggestion = window.currentSuggestions[index];
  if (!suggestion) return;
  
  // Create a new row with the suggested values
  createRow({
    dspb: suggestion.dspb,
    dnase: suggestion.dnase,
    prok: suggestion.prok,
    reactionTime: suggestion.reaction_time
  });
  
  // Scroll to the new row
  const newRow = DOM_ELEMENTS.tableBody.lastElementChild;
  if (newRow) {
    newRow.scrollIntoView({ behavior: 'smooth' });
    
    // Highlight the new row briefly
    newRow.style.backgroundColor = '#e8f5e8';
    setTimeout(() => {
      newRow.style.backgroundColor = '';
    }, 2000);
  }
};

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);
