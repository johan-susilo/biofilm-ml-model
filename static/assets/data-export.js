/**
 * Data Export to CSV
 * 
 * This file lets researchers export their experiment data to CSV files
 * so they can analyze it in Excel, R, Python, or other tools.
 * 
 * What gets exported:
 * - All enzyme ratios and reaction conditions
 * - Predicted biofilm removal percentages
 * - Confidence intervals and uncertainty estimates
 * - Any measured OD600 values
 * 
 * The CSV format is compatible with common analysis software
 * that researchers already use in their workflows.
 */

import { DOM_ELEMENTS } from './config.js';

/**
 * Properly escape a value for CSV format
 * Handles commas, quotes, and newlines in data
 */
function escapeCSVValue(value) {
  if (value === null || value === undefined) {
    return '';
  }
  
  const stringValue = String(value);
  
  // If the value contains comma, quote, or newline, wrap in quotes and escape internal quotes
  if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n') || stringValue.includes('\r')) {
    return '"' + stringValue.replace(/"/g, '""') + '"';
  }
  
  return stringValue;
}

/**
 * Convert current experiment table to CSV format.
 * Normalizes enzyme ratios to sum to 1.0 and includes all relevant data.
 * 
 * Returns:
 *   CSV string with headers and data rows
 */
export function rowsToCsv() {
  try {
    // Define CSV column headers
    const header = [
      'DspB_ratio','DNase_I_ratio','ProK_ratio',
      'pH','Temperature','Total_Volume','biofilm_age_hours','Reaction_Time',
      'Predicted','PI_low','PI_high','Actual_Removal'
    ];
    
    // Get fixed experimental conditions from sidebar inputs
    const pH = parseFloat(document.getElementById('fixed-ph')?.value) || null;
    const temp = parseFloat(document.getElementById('fixed-temp')?.value) || null;
    const totalVol = parseFloat(document.getElementById('total-volume')?.value) || null;
    const age = parseFloat(document.getElementById('biofilm-age')?.value) || null;
    
    // Check if table body exists
    if (!DOM_ELEMENTS.tableBody) {
      console.warn('Table body not found - initializing DOM elements may be needed');
      return header.map(escapeCSVValue).join(',') + '\n';
    }
    
    // Process each experiment row
    const rows = Array.from(DOM_ELEMENTS.tableBody.querySelectorAll('tr')).map((row, index) => {
      try {
        // Extract and normalize enzyme ratios
        let d = parseFloat(row.querySelector('.ratio-dspb')?.value) || 0;
        let n = parseFloat(row.querySelector('.ratio-dnase')?.value) || 0;
        let p = parseFloat(row.querySelector('.ratio-prok')?.value) || 0;
        const s = d + n + p || 1;
        d = d / s; n = n / s; p = p / s;  // Normalize to sum to 1
        
        // Get experimental parameters  
        const rt = parseFloat(row.querySelector('.reaction-time')?.value) || null;
        
        // Extract prediction data
        const predEl = row.querySelector('[data-prediction]');
        const pred = predEl ? parseFloat(predEl.dataset.prediction) : null;
        
        // Extract prediction interval from tooltip text
        let pil = null, pih = null;
        const tip = row.querySelector('.removal-predicted .has-tooltip, .removal-predicted [title]');
        if (tip) {
          const tipText = tip.textContent || tip.getAttribute('title') || '';
          const m = tipText.match(/\[([\d.]+)%\s*-\s*([\d.]+)%\]/);
          if (m) { 
            pil = parseFloat(m[1]) || null; 
            pih = parseFloat(m[2]) || null; 
          }
        }
        
        // Extract actual removal data from OD600 measurements
        const actualEl = row.querySelector('.removal-actual');
        let actualRemoval = null;
        if (actualEl && actualEl.dataset.actual) {
          actualRemoval = parseFloat(actualEl.dataset.actual);
        }
        
        // Create the row data array with proper values
        const rowData = [
          d.toFixed(4),                        // DspB_ratio  
          n.toFixed(4),                        // DNase_I_ratio
          p.toFixed(4),                        // ProK_ratio
          pH !== null ? pH.toString() : '',    // pH
          temp !== null ? temp.toString() : '', // Temperature
          totalVol !== null ? totalVol.toString() : '', // Total_Volume
          age !== null ? age.toString() : '',   // biofilm_age_hours
          rt !== null ? rt.toString() : '',     // Reaction_Time
          pred !== null ? pred.toString() : '', // Predicted
          pil !== null ? pil.toString() : '',   // PI_low
          pih !== null ? pih.toString() : '',   // PI_high
          actualRemoval !== null ? actualRemoval.toString() : '' // Actual_Removal
        ];
        
        // Apply CSV escaping to each value
        return rowData.map(escapeCSVValue);
          
      } catch (rowError) {
        console.error(`Error processing row ${index + 1}:`, rowError);
        // Return empty row with correct number of columns
        return new Array(header.length).fill('');
      }
    });
    
    // Combine header and rows into CSV format
    const csv = [header.map(escapeCSVValue).join(','), ...rows.map(r => r.join(','))].join('\n');
    return csv;
    
  } catch (error) {
    console.error('Error generating CSV:', error);
    return 'Error generating CSV data';
  }
}

/**
 * Download CSV text as a file.
 * Creates a temporary download link and triggers the download.
 */
export function downloadCsv(text, filename='experiments.csv') {
  // Debug: Log the CSV content to console for inspection
  console.log('CSV Export Content:');
  console.log('Length:', text.length);
  console.log('First 200 characters:', text.substring(0, 200));
  console.log('Line count:', text.split('\n').length);
  
  const blob = new Blob([text], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  console.log('CSV file downloaded successfully:', filename);
}
