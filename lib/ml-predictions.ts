import type { Project, RiskLevel } from './demo-data';

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
  explanation?: any;
}

// In-memory cache for the combined dataset
let _cachedProjects: Project[] | null = null;
let _cachedPredictions: MLPrediction[] | null = null;
let _cachedHistory: any = null;

// Normalizer just in case, though combined data should be clean
function normalizeId(id: string | number) {
  return String(id).replace(/[^0-9A-Z-]/gi, '').replace(/^0+/, '');
}

async function loadData() {
  if (!_cachedProjects) {
    const res = await fetch('/nirnay_combined_data.json');
    if (res.ok) {
      const data = await res.json();
      _cachedProjects = data.map((d: any) => ({
        id: String(d.id),
        name: d.name || `Project ${d.id}`,
        ministry: d.ministry || 'Unknown',
        sector: d.sector || 'Unknown',
        state: d.state || 'Unknown',
        originalCost: d.originalCost || 0,
        revisedCost: d.revisedCost || 0,
        expenditure: d.expenditure || 0,
        physicalProgress: d.physicalProgress || 0,
        financialProgress: d.financialProgress || 0,
        riskScore: d.riskScore || 0,
        costRisk: d.costRiskIndicator || 0,
        delayRisk: Math.round((d.probabilityDelayed || 0) * 100),
        riskLevel: (d.predictedRiskLevel || 'Low') as RiskLevel,
        status: d.status || 'Under Implementation',
        lifecycle: (d.physicalProgress === 100 || String(d.status).toLowerCase() === 'completed') ? (d.timeOverrunMonths > 0 ? 'Completed - Delayed' : 'Completed on Schedule') : 'Active / Ongoing',
        timeOverrunMonths: d.timeOverrunMonths || 0,
        primaryDriver: d.primaryDriver || 'Multiple factors',
        plannedCompletion: d.originalCompletion || 'N/A',
        expectedCompletion: d.anticipatedCompletion || 'N/A',
        implementingAgency: d.agency || 'N/A',
        riskHistory: [] // Will be populated dynamically if needed
      }));
      
      _cachedPredictions = data.map((d: any) => ({
        project_id: String(d.id),
        risk_level_ml: (d.predictedRiskLevel || 'Low') as RiskLevel,
        probability_delayed_6m: d.probabilityDelayed || 0,
        risk_confidence: d.probabilityDelayed || 0,
        probability_critical: d.probabilityCritical || 0,
        probability_high: d.probabilityHigh || 0,
        probability_medium: d.probabilityMedium || 0,
        probability_low: d.probabilityLow || 0,
        top_risk_driver_1: d.topDriver1 || d.primaryDriver || '',
        top_risk_driver_2: d.topDriver2 || '',
        top_risk_driver_3: d.topDriver3 || '',
        explanation: d.explanation
      }));
    } else {
      _cachedProjects = [];
      _cachedPredictions = [];
    }
  }
}

export async function getMLPredictions(): Promise<MLPrediction[]> {
  await loadData();
  return _cachedPredictions || [];
}

export async function getEnrichedProjects(): Promise<Project[]> {
  await loadData();
  
  console.log(`DATA LOAD DEBUG
----------------`);
  console.log(`Project records loaded: ${_cachedProjects?.length || 0}`);
  console.log(`Prediction records loaded: ${_cachedPredictions?.length || 0}`);
  console.log(`Historical records loaded: ${Object.keys(_cachedHistory || {}).length}`);
  console.log(`Successful project/prediction joins: ${_cachedProjects?.filter(p => _cachedPredictions?.some(pr => pr.project_id === p.id)).length || 0}`);
  console.log(`Projects with missing prediction: ${_cachedProjects?.filter(p => !_cachedPredictions?.some(pr => pr.project_id === p.id)).length || 0}`);
  console.log(`Valid risk scores: ${_cachedProjects?.filter(p => p.riskScore > 0).length || 0}`);
  console.log(`Valid delay probabilities: ${_cachedProjects?.filter(p => p.delayRisk > 0).length || 0}`);
  console.log(`Critical: ${_cachedProjects?.filter(p => p.riskLevel === 'Critical').length || 0}`);
  console.log(`High: ${_cachedProjects?.filter(p => p.riskLevel === 'High').length || 0}`);
  console.log(`Medium: ${_cachedProjects?.filter(p => p.riskLevel === 'Medium').length || 0}`);
  console.log(`Low: ${_cachedProjects?.filter(p => p.riskLevel === 'Low').length || 0}`);
  console.log(`Early warnings: ${_cachedProjects?.filter(p => p.riskLevel === 'Critical' && p.delayRisk > 75).length || 0}`);
  return _cachedProjects || [];
}

export async function getEnrichedProjectById(id: string): Promise<Project | undefined> {
  const all = await getEnrichedProjects();
  return all.find(p => p.id === String(id) || normalizeId(p.id) === normalizeId(id));
}

export async function getMLPredictionByProjectId(projectId: string): Promise<MLPrediction | undefined> {
  const preds = await getMLPredictions();
  return preds.find(p => p.project_id === String(projectId) || normalizeId(p.project_id) === normalizeId(projectId));
}

export async function getRiskStatistics() {
  const projects = await getEnrichedProjects();
  const counts = { Critical: 0, High: 0, Medium: 0, Low: 0 };
  for (const p of projects) {
    if (counts[p.riskLevel] !== undefined) {
      counts[p.riskLevel]++;
    }
  }
  return counts;
}

export async function getMLWarnings() {
  const projects = await getEnrichedProjects();
  const preds = await getMLPredictions();
  
  // Find critical projects with high delay risk
  const criticals = projects.filter(p => p.riskLevel === 'Critical' && p.delayRisk > 75).slice(0, 8);
  
  return criticals.map(p => {
    const pred = preds.find(pr => pr.project_id === p.id);
    return {
      title: `${p.primaryDriver} detected`,
      project: p.name,
      detail: `Model flags a ${p.delayRisk}% probability of schedule delay driven by ${p.primaryDriver.toLowerCase()}.`,
      metric: `Probability: ${p.delayRisk}%`,
      id: p.id,
      severity: p.riskLevel
    };
  });
}

export async function getProjectHistoryData(id: string) {
  if (!_cachedHistory) {
    const res = await fetch('/nirnay_project_history.json');
    if (res.ok) _cachedHistory = await res.json();
    else _cachedHistory = {};
  }
  return _cachedHistory[normalizeId(id)] || [];
}

export async function getPortfolioHistoryData() {
  if (!_cachedHistory) {
    const res = await fetch('/nirnay_project_history.json');
    if (res.ok) _cachedHistory = await res.json();
    else _cachedHistory = {};
  }
  return _cachedHistory;
}


export function filterProjects(projects: Project[], query: string, ministry: string, sector: string, risk: string, status: string = 'All Statuses'): Project[] {
  const normalize = (v: any) => String(v || '').trim().toLowerCase();
  const nq = normalize(query);
  const nm = normalize(ministry);
  const ns = normalize(sector);
  const nr = normalize(risk);
  const nst = normalize(status);
  
  return projects.filter(p => {
    const pq = normalize(p.name) + ' ' + normalize(p.id);
    const pm = normalize(p.ministry);
    const ps = normalize(p.sector);
    const pr = normalize(p.riskLevel);
    const pst = normalize(p.lifecycle || p.status);
    
    const matchQ = !nq || pq.includes(nq);
    const matchM = nm === 'all ministries' || nm === 'all' || pm === nm;
    const matchS = ns === 'all sectors' || ns === 'all' || ps === ns;
    const matchR = nr === 'all risk levels' || nr === 'all' || pr === nr;
    const matchSt = nst === 'all statuses' || nst === 'all' || pst === nst;
    
    return matchQ && matchM && matchS && matchR && matchSt;
  });
}

export function getUniqueMinistries(projects: Project[]): string[] {
  return Array.from(new Set(projects.map(p => p.ministry).filter(Boolean))).sort();
}
export function getUniqueSectors(projects: Project[]): string[] {
  return Array.from(new Set(projects.map(p => p.sector).filter(Boolean))).sort();
}
export function getUniqueStatuses(projects: Project[]): string[] {
  return Array.from(new Set(projects.map(p => p.lifecycle).filter(Boolean))).sort();
}
