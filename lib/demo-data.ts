export type RiskLevel = 'Critical' | 'High' | 'Medium' | 'Low'
export interface RiskPoint { month: string; value: number }
export interface Project {
  id: string; name: string; ministry: string; sector: string; state: string
  originalCost: number; revisedCost: number; expenditure: number
  physicalProgress: number; financialProgress: number; riskScore: number
  costRisk: number; delayRisk: number; riskLevel: RiskLevel; status: string
  primaryDriver: string; plannedCompletion: string; expectedCompletion: string
  implementingAgency: string; riskHistory: RiskPoint[]; lifecycle?: string; timeOverrunMonths?: number;
}

export const projects: Project[] = [
  { id:'NIR-2026-001', name:'Eastern Dedicated Freight Corridor', ministry:'Railways', sector:'Transport', state:'Uttar Pradesh', originalCost:81500, revisedCost:93200, expenditure:64800, physicalProgress:72, financialProgress:70, riskScore:91, costRisk:78, delayRisk:91, riskLevel:'Critical', status:'Under Implementation', primaryDriver:'Milestone slippage', plannedCompletion:'Dec 2026', expectedCompletion:'Aug 2027', implementingAgency:'Dedicated Freight Corridor Corporation', riskHistory:[{month:'Jan',value:42},{month:'Feb',value:46},{month:'Mar',value:51},{month:'Apr',value:59},{month:'May',value:68},{month:'Jun',value:77},{month:'Jul',value:91}] },
  { id:'NIR-2026-014', name:'Mumbai–Ahmedabad High Speed Rail', ministry:'Railways', sector:'Transport', state:'Maharashtra', originalCost:110000, revisedCost:118400, expenditure:56300, physicalProgress:48, financialProgress:47, riskScore:78, costRisk:64, delayRisk:82, riskLevel:'High', status:'Under Implementation', primaryDriver:'Physical progress deviation', plannedCompletion:'Mar 2028', expectedCompletion:'Nov 2028', implementingAgency:'National High Speed Rail Corporation', riskHistory:[{month:'Jan',value:38},{month:'Feb',value:41},{month:'Mar',value:49},{month:'Apr',value:55},{month:'May',value:63},{month:'Jun',value:71},{month:'Jul',value:78}] },
  { id:'NIR-2026-023', name:'River Basin Development Project', ministry:'Jal Shakti', sector:'Water Resources', state:'Bihar', originalCost:26400, revisedCost:28900, expenditure:18500, physicalProgress:64, financialProgress:68, riskScore:74, costRisk:59, delayRisk:77, riskLevel:'High', status:'Under Implementation', primaryDriver:'Expenditure-progress mismatch', plannedCompletion:'Jun 2027', expectedCompletion:'Jan 2028', implementingAgency:'National Water Development Agency', riskHistory:[{month:'Jan',value:36},{month:'Feb',value:43},{month:'Mar',value:47},{month:'Apr',value:52},{month:'May',value:61},{month:'Jun',value:69},{month:'Jul',value:74}] },
  { id:'NIR-2026-031', name:'National Highway Development Package', ministry:'Road Transport & Highways', sector:'Transport', state:'Rajasthan', originalCost:47200, revisedCost:49800, expenditure:32100, physicalProgress:78, financialProgress:75, riskScore:54, costRisk:42, delayRisk:57, riskLevel:'Medium', status:'Under Implementation', primaryDriver:'Contractor performance', plannedCompletion:'Oct 2026', expectedCompletion:'Feb 2027', implementingAgency:'National Highways Authority of India', riskHistory:[{month:'Jan',value:31},{month:'Feb',value:34},{month:'Mar',value:39},{month:'Apr',value:43},{month:'May',value:48},{month:'Jun',value:52},{month:'Jul',value:54}] },
  { id:'NIR-2026-045', name:'Integrated Power Transmission Project', ministry:'Power', sector:'Energy', state:'Madhya Pradesh', originalCost:18800, revisedCost:20100, expenditure:14200, physicalProgress:83, financialProgress:81, riskScore:47, costRisk:71, delayRisk:39, riskLevel:'Medium', status:'Under Implementation', primaryDriver:'Cost acceleration', plannedCompletion:'Sep 2026', expectedCompletion:'Dec 2026', implementingAgency:'Power Grid Corporation of India', riskHistory:[{month:'Jan',value:28},{month:'Feb',value:30},{month:'Mar',value:33},{month:'Apr',value:38},{month:'May',value:41},{month:'Jun',value:45},{month:'Jul',value:47}] },
  { id:'NIR-2026-052', name:'Regional Water Infrastructure Project', ministry:'Jal Shakti', sector:'Water Resources', state:'Odisha', originalCost:12600, revisedCost:12900, expenditure:10100, physicalProgress:88, financialProgress:87, riskScore:27, costRisk:18, delayRisk:31, riskLevel:'Low', status:'Under Implementation', primaryDriver:'Seasonal variation', plannedCompletion:'May 2026', expectedCompletion:'Jun 2026', implementingAgency:'State Water Resources Department', riskHistory:[{month:'Jan',value:24},{month:'Feb',value:25},{month:'Mar',value:26},{month:'Apr',value:25},{month:'May',value:28},{month:'Jun',value:27},{month:'Jul',value:27}] },
]
export const riskCounts = { Critical:86, High:214, Medium:641, Low:1040 }
export const ministries = ['All Ministries','Railways','Jal Shakti','Road Transport & Highways','Power']
export const sectors = ['All Sectors','Transport','Water Resources','Energy']
export const warnings = [
  { title:'Progress stagnation detected', project:'Eastern Dedicated Freight Corridor', detail:'Physical progress has remained below the expected trajectory for three consecutive reporting periods.', metric:'Risk increased: 64 → 79', id:'NIR-2026-001' },
  { title:'Cost acceleration detected', project:'Integrated Power Transmission Project', detail:'Recent expenditure trends indicate increased probability of cost escalation.', metric:'Cost risk: 71%', id:'NIR-2026-045' },
  { title:'Milestone slippage detected', project:'Regional Water Infrastructure Project', detail:'Two critical milestones have exceeded their expected completion dates.', metric:'Schedule risk: High', id:'NIR-2026-052' },
]
export const formatCr = (n:number) => `₹${n.toLocaleString('en-IN')} Cr`
export const riskClass = (level:RiskLevel) => ({Critical:'risk-critical',High:'risk-high',Medium:'risk-medium',Low:'risk-low'}[level])
export const projectById = (id:string) => projects.find(p => p.id === id) ?? projects[0]
