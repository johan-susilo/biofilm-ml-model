/**
 * Data Visualizations
 * 
 * This file creates all the charts and graphs that help researchers
 * understand their data and results:
 * 
 * - Feature importance charts (which factors matter most)
 * - Ternary plots (3D visualization of enzyme combinations)
 * - Summary statistics and trends
 * 
 * Uses Plotly.js to create interactive, publication-quality charts
 * that researchers can explore and export for their papers.
 */

import { API_URL } from './config.js';
import { fetchFeatureImportance } from './api-services.js';
import { proportionsToIntegers } from './calculations.js';
import { updateSummaryMetrics, updatePredictionAnalysis } from './ui-components.js';

/**
 * Render a full-width feature-importance plot, focusing on core inputs.
 */
export async function renderCoreFeatureImportancePlot() {
  try {
    const data = await fetchFeatureImportance();
    if (!data) return;
    
    const { features, values } = data;
    const card = document.getElementById('plot-feature-importance')?.closest('.card');
    if (card) card.style.display = 'block';
    
    const layout = { 
      margin: { l: 180, r: 60, t: 20, b: 80 }, // Much more left margin for feature names
      height: 600, // Increased height for better readability
      font: { family: 'Inter, sans-serif', size: 13 },
      plot_bgcolor: '#fafafa',
      paper_bgcolor: '#ffffff',
      xaxis: {
        title: {
          text: 'Importance Score',
          font: { size: 16, family: 'Inter, sans-serif', color: '#374151' },
          standoff: 20
        },
        showgrid: true,
        gridcolor: '#e5e7eb',
        gridwidth: 1,
        tickfont: { size: 13, color: '#374151' },
        showline: true,
        linecolor: '#d1d5db',
        linewidth: 1
      },
      yaxis: {
        title: '',
        showgrid: false,
        tickfont: { size: 13, color: '#374151', family: 'Inter, sans-serif' },
        showline: false,
        ticklen: 8,
        tickcolor: '#d1d5db'
      },
      autosize: true,
      bargap: 0.4, // Increase space between bars
    };
    
    const plotData = [{ 
      type: 'bar', 
      x: values.slice().reverse(), 
      y: features.slice().reverse(), 
      orientation: 'h', 
      marker: { 
        color: '#a8c899',
        line: { color: '#1a472a', width: 1.5 },
        opacity: 0.8
      },
      hovertemplate: '<b>%{y}</b><br>Importance: <b>%{x:.3f}</b><extra></extra>',
      hoverlabel: {
        bgcolor: '#1a472a',
        bordercolor: '#a8c899',
        font: { color: 'white', family: 'Inter, sans-serif', size: 12 }
      }
    }];
    
    const config = { 
      responsive: true, 
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: 'feature_importance',
        height: 550,
        width: 1200,
        scale: 1
      }
    };
    
    // Create the plot
    Plotly.newPlot('plot-feature-importance', plotData, layout, config);
    
    // Add window resize listener to ensure proper responsiveness
    const resizeHandler = () => {
      Plotly.Plots.resize('plot-feature-importance');
    };
    
    // Remove existing listener if any, then add new one
    window.removeEventListener('resize', resizeHandler);
    window.addEventListener('resize', resizeHandler);
    
  } catch (e) {
    console.error('Feature importance plot failed:', e);
  }
}

/**
 * Update all visualizations in the application
 */
export function updateAllVisualizations() {
  updateSummaryMetrics();
  updatePredictionAnalysis();
  updateTernaryPredictedPlot();
  updateTernaryActualPlot();
}

/**
 * Update ternary plot showing predicted removal rates
 */
function updateTernaryPredictedPlot() {
  const rows = Array.from(document.querySelectorAll('#table-body tr'));
  if (rows.length === 0) return;
  
  // Collect all valid data points
  const dataPoints = [];
  rows.forEach(row => {
    const predEl = row.querySelector('[data-prediction]');
    const pred = predEl ? parseFloat(predEl.dataset.prediction) : NaN;
    
    if (isNaN(pred)) return;
    
    const dspb = parseFloat(row.querySelector('.ratio-dspb')?.value) || 0;
    const dnase = parseFloat(row.querySelector('.ratio-dnase')?.value) || 0;
    const prok = parseFloat(row.querySelector('.ratio-prok')?.value) || 0;
    
    const ints = proportionsToIntegers({ DspB: dspb, DNase: dnase, ProK: prok }, 100);
    
    dataPoints.push({
      a: ints.DspB,
      b: ints.DNase, 
      c: ints.ProK,
      prediction: pred
    });
  });
  
  if (dataPoints.length === 0) {
    // Clear the plot if no prediction data
    if (document.getElementById('ternary-predicted-plot')) {
      Plotly.purge('ternary-predicted-plot');
    }
    return;
  }
  
  // Create single trace with all points
  const plotData = [{
    type: 'scatterternary',
    mode: 'markers+text',
    a: dataPoints.map(d => d.a),
    b: dataPoints.map(d => d.b),
    c: dataPoints.map(d => d.c),
    text: dataPoints.map(d => `${d.prediction.toFixed(1)}%`),
    textposition: 'top center',
    textfont: { size: 12, color: 'white', family: 'Inter, sans-serif' },
    marker: {
      size: 16, // Increased marker size for better visibility
      color: dataPoints.map(d => d.prediction),
      colorscale: 'Viridis',
      showscale: true,
      colorbar: { 
        title: 'Predicted Removal %',
        titlefont: { size: 12, family: 'Inter, sans-serif' },
        tickfont: { size: 10, family: 'Inter, sans-serif' }
      },
      line: { width: 2, color: 'white' }, // Thicker border for better contrast
    },
    hovertemplate: 'DspB: %{a}%<br>DNase: %{b}%<br>ProK: %{c}%<br>Predicted: %{text}<extra></extra>',
  }];
  
  const layout = {
    ternary: {
      sum: 100,
      aaxis: { 
        title: {
          text: 'Dispersin B (%)',
          font: { size: 14, family: 'Inter, sans-serif', color: '#374151' }
        },
        tickfont: { size: 11, family: 'Inter, sans-serif' }
      },
      baxis: { 
        title: {
          text: 'DNase I (%)',
          font: { size: 14, family: 'Inter, sans-serif', color: '#374151' }
        },
        tickfont: { size: 11, family: 'Inter, sans-serif' }
      },
      caxis: { 
        title: {
          text: 'Proteinase K (%)',
          font: { size: 14, family: 'Inter, sans-serif', color: '#374151' }
        },
        tickfont: { size: 11, family: 'Inter, sans-serif' }
      },
    },
    margin: { l: 80, r: 80, t: 100, b: 80 }, // Increased margins for better spacing
    font: { family: 'Inter, sans-serif', size: 12 },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
    autosize: true // Enable automatic resizing
  };
  
  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    toImageButtonOptions: {
      format: 'png',
      filename: 'predicted_removal_ternary',
      height: 500,
      width: 600,
      scale: 1
    }
  };
  
  Plotly.newPlot('ternary-predicted-plot', plotData, layout, config);
  
  // Add resize handler for responsiveness
  const resizeHandler = () => {
    Plotly.Plots.resize('ternary-predicted-plot');
  };
  window.removeEventListener('resize', resizeHandler);
  window.addEventListener('resize', resizeHandler);
}

/**
 * Update ternary plot showing actual removal rates from OD600 measurements
 */
function updateTernaryActualPlot() {
  const rows = Array.from(document.querySelectorAll('#table-body tr'));
  if (rows.length === 0) return;
  
  // Collect all valid data points
  const dataPoints = [];
  rows.forEach(row => {
    const actualEl = row.querySelector('.removal-actual');
    const actual = actualEl?.dataset.actual ? parseFloat(actualEl.dataset.actual) : NaN;
    
    if (isNaN(actual)) return;
    
    const dspb = parseFloat(row.querySelector('.ratio-dspb')?.value) || 0;
    const dnase = parseFloat(row.querySelector('.ratio-dnase')?.value) || 0;
    const prok = parseFloat(row.querySelector('.ratio-prok')?.value) || 0;
    
    const ints = proportionsToIntegers({ DspB: dspb, DNase: dnase, ProK: prok }, 100);
    
    dataPoints.push({
      a: ints.DspB,
      b: ints.DNase,
      c: ints.ProK,
      actual: actual
    });
  });
  
  if (dataPoints.length === 0) {
    // Clear the plot if no actual data
    if (document.getElementById('ternary-actual-plot')) {
      Plotly.purge('ternary-actual-plot');
    }
    return;
  }
  
  // Create single trace with all points
  const plotData = [{
    type: 'scatterternary',
    mode: 'markers+text',
    a: dataPoints.map(d => d.a),
    b: dataPoints.map(d => d.b),
    c: dataPoints.map(d => d.c),
    text: dataPoints.map(d => `${d.actual.toFixed(1)}%`),
    textposition: 'top center',
    textfont: { size: 12, color: 'white', family: 'Inter, sans-serif' },
    marker: {
      size: 16, // Increased marker size for better visibility
      color: dataPoints.map(d => d.actual),
      colorscale: 'Plasma',
      showscale: true,
      colorbar: { 
        title: 'Actual Removal %',
        titlefont: { size: 12, family: 'Inter, sans-serif' },
        tickfont: { size: 10, family: 'Inter, sans-serif' }
      },
      line: { width: 2, color: 'white' }, // Thicker border for better contrast
    },
    hovertemplate: 'DspB: %{a}%<br>DNase: %{b}%<br>ProK: %{c}%<br>Actual: %{text}<extra></extra>',
  }];
  
  const layout = {
    ternary: {
      sum: 100,
      aaxis: { 
        title: {
          text: 'Dispersin B (%)',
          font: { size: 14, family: 'Inter, sans-serif', color: '#374151' }
        },
        tickfont: { size: 11, family: 'Inter, sans-serif' }
      },
      baxis: { 
        title: {
          text: 'DNase I (%)',
          font: { size: 14, family: 'Inter, sans-serif', color: '#374151' }
        },
        tickfont: { size: 11, family: 'Inter, sans-serif' }
      },
      caxis: { 
        title: {
          text: 'Proteinase K (%)',
          font: { size: 14, family: 'Inter, sans-serif', color: '#374151' }
        },
        tickfont: { size: 11, family: 'Inter, sans-serif' }
      },
    },
    margin: { l: 80, r: 80, t: 100, b: 80 }, // Increased margins for better spacing
    font: { family: 'Inter, sans-serif', size: 12 },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
    autosize: true // Enable automatic resizing
  };
  
  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    toImageButtonOptions: {
      format: 'png',
      filename: 'actual_removal_ternary',
      height: 500,
      width: 600,
      scale: 1
    }
  };
  
  Plotly.newPlot('ternary-actual-plot', plotData, layout, config);
  
  // Add resize handler for responsiveness
  const resizeHandler = () => {
    Plotly.Plots.resize('ternary-actual-plot');
  };
  window.removeEventListener('resize', resizeHandler);
  window.addEventListener('resize', resizeHandler);
}
