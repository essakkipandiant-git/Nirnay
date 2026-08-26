import Papa from 'papaparse';
import { Project, projects as demoProjects, RiskLevel } from './demo-data';

export interface MLPrediction {
  project_id: string;
  risk_level_ml: RiskLevel;
  probability_delayed_6m: number;
  risk_confidence: number;
  probability_critical: number;
  probability_high: number;
  probability_medium: number;
  probability_low: number;
  top_risk_driver_1: string;
  top_risk_driver_2: string;
  top_risk_driver_3: string;
}

// ---------------------------------------------------------
// PREDICTION ADAPTER PATTERN
// ---------------------------------------------------------

export interface PredictionAdapter {
  getMLPredictions(): Promise<MLPrediction[]>;
}

export class CSVPredictionProvider implements PredictionAdapter {
  async getMLPredictions(): Promise<MLPrediction[]> {
    // DO NOT read from local filesystem in browser. Fetch from public static path.
    const response = await fetch('/nirnay_ml_predictions.csv');
    if (!response.ok) {
      console.warn('Failed to fetch ML predictions CSV. Returning empty.');
      return [];
    }
    const csvText = await response.text();
    const parsed = Papa.parse<MLPrediction>(csvText, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
    });
    return parsed.data;
  }
}

// Active provider instance (can be swapped for APIPredictionProvider later)
const activeProvider: PredictionAdapter = new CSVPredictionProvider();

// In-memory cache
let _predictionsCache: MLPrediction[] | null = null;

// ---------------------------------------------------------
// EXPOSED SERVICE FUNCTIONS
// ---------------------------------------------------------

export async function getMLPredictions(): Promise<MLPrediction[]> {
  if (!_predictionsCache) {
    _predictionsCache = await activeProvider.getMLPredictions();
  }
  return _predictionsCache;
}

/**
 * Clean up IDs to handle mismatches (e.g. NIR-2026-001 vs 2026001 or 14)
 */
function normalizeId(id: string | number) {
  return String(id).replace(/[^0-9]/g, '').replace(/^0+/, '');
}

export async function getMLPredictionByProjectId(projectId: string): Promise<MLPrediction | undefined> {
  const preds = await getMLPredictions();
  const normalizedTarget = normalizeId(projectId);
  return preds.find((p) => {
    // If exact match (in case ids match exactly)
    if (String(p.project_id) === projectId) return true;
    // Fallback fuzzy match for demo data IDs like "NIR-2026-014" vs "14"
    if (normalizeId(p.project_id) === normalizedTarget) return true;
    return false;
  });
}

export async function getRiskStatistics() {
  const preds = await getMLPredictions();
  const counts = { Critical: 0, High: 0, Medium: 0, Low: 0 };
  
  for (const p of preds) {
    if (counts[p.risk_level_ml] !== undefined) {
      counts[p.risk_level_ml]++;
    }
  }
  return counts;
}

// ---------------------------------------------------------
// FRONTEND INTEGRATION HELPERS
// ---------------------------------------------------------

/**
 * Deterministic Risk Score calculation from probabilities (0-100 scale)
 */
export function computeRiskScore(prediction: MLPrediction | undefined): number {
  if (!prediction) return 0;
  return Math.round(prediction.probability_delayed_6m * 100);
}

/**
 * Transforms demo projects by injecting REAL ML predictions
 */
export async function getEnrichedProjects(): Promise<Project[]> {
  const predictions = await getMLPredictions();
  
  // Clone demo projects
  const enriched: Project[] = JSON.parse(JSON.stringify(demoProjects));

  for (const project of enriched) {
    const mlPred = predictions.find(p => 
      String(p.project_id) === project.id || 
      normalizeId(p.project_id) === normalizeId(project.id)
    );

    if (mlPred) {
      // OVERWRITE demo values with ML predictions
      project.riskLevel = mlPred.risk_level_ml;
      project.riskScore = computeRiskScore(mlPred);
      project.delayRisk = Math.round(mlPred.probability_delayed_6m * 100);
      project.primaryDriver = mlPred.top_risk_driver_1 || 'Unspecified';
      
      // We leave costRisk intact as instructed: it represents current exposure, not ML probability.
    } else {
      // MISSING PREDICTION HANDLING
      project.riskLevel = 'Prediction unavailable' as any;
      project.riskScore = 0;
      project.delayRisk = 0;
      project.primaryDriver = 'Prediction unavailable';
    }
  }
  
  return enriched;
}

export async function getEnrichedProjectById(id: string): Promise<Project> {
  const all = await getEnrichedProjects();
  return all.find(p => p.id === id) || all[0];
}

/**
 * Generate Early Warnings derived dynamically from ML Predictions
 */
export async function getMLWarnings() {
  const preds = await getMLPredictions();
  const criticals = preds.filter(p => p.risk_level_ml === 'Critical').slice(0, 5);
  
  // Generate warnings based on top risk drivers
  return criticals.map(p => {
    // Find matching demo project for names (in real app, we'd have the name)
    const demoMatch = demoProjects.find(dp => normalizeId(dp.id) === normalizeId(p.project_id));
    const name = demoMatch ? demoMatch.name : `Project ${p.project_id}`;
    const id = demoMatch ? demoMatch.id : String(p.project_id);
    
    return {
      title: `${p.top_risk_driver_1 || 'Critical Risk'} detected`,
      project: name,
      detail: `Model has flagged a ${Math.round(p.probability_delayed_6m * 100)}% probability of schedule delay driven by ${p.top_risk_driver_1?.toLowerCase() || 'multiple factors'}.`,
      metric: `Probability: ${Math.round(p.probability_delayed_6m * 100)}% | Conf: ${Math.round(p.risk_confidence * 100)}%`,
      id: id,
    };
  });
}
