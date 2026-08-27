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
let _dataLoadPromise: Promise<void> | null = null;
let _historyLoadPromise: Promise<void> | null = null;

const asNumber = (value: unknown): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const asText = (value: unknown, fallback = 'Unknown'): string => {
  const text = String(value ?? '').trim();
  return text || fallback;
};

// Normalizer just in case, though combined data should be clean
function normalizeId(id: string | number) {
  return String(id).replace(/[^0-9A-Z-]/gi, '').replace(/^0+/, '');
}

async function loadData() {
  if (_cachedProjects) return;
  if (!_dataLoadPromise) {
    _dataLoadPromise = (async () => {
      const res = await fetch('/nirnay_combined_data.json');
      if (!res.ok) throw new Error(`Unable to load project data (${res.status}).`);

      const data: unknown = await res.json();
      if (!Array.isArray(data)) throw new Error('Project data has an invalid format.');

      const records = data.filter((record: any) => record && record.id != null);
      _cachedProjects = records.map((d: any) => {
        const physicalProgress = asNumber(d.physicalProgress);
        const timeOverrunMonths = asNumber(d.timeOverrunMonths);
        const status = asText(d.status, 'Under Implementation');
        const probabilityDelayed = Math.max(0, Math.min(1, asNumber(d.probabilityDelayed)));

        return {
          id: String(d.id),
          name: asText(d.name, `Project ${d.id}`),
          ministry: asText(d.ministry),
          sector: asText(d.sector),
          state: asText(d.state),
          originalCost: asNumber(d.originalCost),
          revisedCost: asNumber(d.revisedCost),
          expenditure: asNumber(d.expenditure),
          physicalProgress,
          financialProgress: asNumber(d.financialProgress),
          riskScore: asNumber(d.riskScore),
          costRisk: asNumber(d.costRiskIndicator),
          delayRisk: Math.round(probabilityDelayed * 100),
          riskLevel: (d.predictedRiskLevel || 'Low') as RiskLevel,
          status,
          lifecycle: physicalProgress === 100 || status.toLowerCase() === 'completed'
            ? (timeOverrunMonths > 0 ? 'Completed - Delayed' : 'Completed on Schedule')
            : 'Active / Ongoing',
          timeOverrunMonths,
          primaryDriver: asText(d.primaryDriver, 'Multiple factors'),
          plannedCompletion: asText(d.originalCompletion, 'N/A'),
          expectedCompletion: asText(d.anticipatedCompletion, 'N/A'),
          implementingAgency: asText(d.agency, 'N/A'),
          riskHistory: []
        };
      });

      _cachedPredictions = records.map((d: any) => ({
        project_id: String(d.id),
        risk_level_ml: (d.predictedRiskLevel || 'Low') as RiskLevel,
        probability_delayed_6m: Math.max(0, Math.min(1, asNumber(d.probabilityDelayed))),
        risk_confidence: Math.max(0, Math.min(1, asNumber(d.probabilityDelayed))),
        probability_critical: Math.max(0, Math.min(1, asNumber(d.probabilityCritical))),
        probability_high: Math.max(0, Math.min(1, asNumber(d.probabilityHigh))),
        probability_medium: Math.max(0, Math.min(1, asNumber(d.probabilityMedium))),
        probability_low: Math.max(0, Math.min(1, asNumber(d.probabilityLow))),
        top_risk_driver_1: asText(d.topDriver1 || d.primaryDriver, ''),
        top_risk_driver_2: asText(d.topDriver2, ''),
        top_risk_driver_3: asText(d.topDriver3, ''),
        explanation: d.explanation
      }));
    })().catch((error) => {
      _dataLoadPromise = null;
      _cachedProjects = [];
      _cachedPredictions = [];
      console.error('NIRNAY project data loading failed.', error);
    });
  }
  await _dataLoadPromise;
}

export async function getMLPredictions(): Promise<MLPrediction[]> {
  await loadData();
  return _cachedPredictions || [];
}

export async function getEnrichedProjects(): Promise<Project[]> {
  await loadData();
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
  await loadHistory();
  return _cachedHistory[normalizeId(id)] || [];
}

export async function getPortfolioHistoryData() {
  await loadHistory();
  return _cachedHistory;
}

async function loadHistory() {
  if (_cachedHistory) return;
  if (!_historyLoadPromise) {
    _historyLoadPromise = (async () => {
      const res = await fetch('/nirnay_project_history.json');
      if (!res.ok) throw new Error(`Unable to load project history (${res.status}).`);

      const history: unknown = await res.json();
      if (!history || Array.isArray(history) || typeof history !== 'object') {
        throw new Error('Project history has an invalid format.');
      }
      _cachedHistory = history;
    })().catch((error) => {
      _historyLoadPromise = null;
      _cachedHistory = {};
      console.error('NIRNAY project history loading failed.', error);
    });
  }
  await _historyLoadPromise;
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
  return Array.from(new Set(projects.map(p => p.lifecycle).filter((value): value is string => Boolean(value)))).sort();
}
