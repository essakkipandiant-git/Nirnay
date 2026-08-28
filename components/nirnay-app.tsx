'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Bell, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Download, Filter, LayoutDashboard, Menu, Moon, Search, Settings, ShieldCheck, SlidersHorizontal, Sparkles, Sun, Table2, X, Zap } from 'lucide-react'
import { formatCr, ministries, sectors, type Project, type RiskLevel } from '@/lib/demo-data'
import { getEnrichedProjects, getRiskStatistics, getMLWarnings, getMLPredictionByProjectId, getProjectHistoryData, getPortfolioHistoryData, filterProjects, getUniqueMinistries, getUniqueSectors, getUniqueStatuses } from '@/lib/ml-predictions'

const navGroups = [
  { label:'OVERVIEW', collapsible:false, items:[['Dashboard','/dashboard','dashboard',LayoutDashboard,'active'],['Model & Data','/model-performance','model-data',Table2,'active']] },
  { label:'MONITORING', collapsible:false, items:[['Projects','/projects','projects',Table2,'active'],['Early Warnings','/warnings','warnings',Zap,'active'],['Interventions','/interventions','interventions',SlidersHorizontal,'active']] },
  { label:'ANALYTICS', collapsible:true, items:[['Cost Analytics','/analytics/cost','cost',Table2],['Schedule Analytics','/analytics/schedule','schedule',Table2],['Sector Benchmarking','/analytics/sectors','sectors',Table2],['Ministry Analytics','/analytics/ministries','ministries',Table2],['Historical Analysis','/analytics/history','history',Table2]] },
  { label:'INTELLIGENCE', collapsible:true, items:[['Ask NIRNAY','/intelligence','ask-nirnay',Sparkles,'active'],['Project Intelligence','/intelligence','intelligence',Sparkles],['Similar Projects','/projects?view=similar','similar',Table2]] },
  { label:'ADMINISTRATION', collapsible:true, items:[['Data Quality','/data-quality','data-quality',ShieldCheck],['Model Performance','/model-performance','models',Zap],['Scenario Analysis','/simulation','simulation',SlidersHorizontal],['Settings','/settings','settings',Settings]] },
]

export const riskClass = (level: string) => ({'Critical':'risk-critical','High':'risk-high','Medium':'risk-medium','Low':'risk-low'}[level] || 'risk-unavailable')

function RiskBadge({ level, score }:{level: string;score:number}) { 
  if (level === 'Prediction unavailable') return <span className="risk-badge risk-unavailable">Prediction unavailable</span>;
  return <span className={`risk-badge ${riskClass(level)}`}><span className="risk-dot" />{level.toUpperCase()} — {score}</span> 
}
function Card({ title, subtitle, children, className='' }:{title:string;subtitle?:string;children:React.ReactNode;className?:string}) { return <section className={`panel ${className}`}><div className="panel-head"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div></div>{children}</section> }
function PageHeader({ title, subtitle }:{title:string;subtitle:string}) { return <div className="page-header"><div className="breadcrumbs"><Link href="/">NIRNAY</Link><span>/</span><span>{title}</span></div><h1>{title}</h1><p>{subtitle}</p></div> }
function Bar({ value, color='blue' }:{value:number;color?:string}) { return <div className="bar-track"><div className={`bar-fill ${color}`} style={{width:`${value}%`}} /></div> }
function ExportButton({ rows=[] }:{rows?:Project[]}) { const exportCsv=()=>{ const csv=['Project,Ministry,Sector,Risk,Status',...rows.map(p=>[p.name,p.ministry,p.sector,p.riskScore,p.status].map(v=>`"${v}"`).join(','))].join('\n'); const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='nirnay-projects.csv';a.click() }; return <button className="button secondary" onClick={exportCsv}><Download size={15}/> Export</button> }
function Sidebar({open,onClose}:{open:boolean;onClose:()=>void}) { const path=usePathname(); const [comingSoon,setComingSoon]=useState(false); const [expanded,setExpanded]=useState<Record<string,boolean>>({}); const showComingSoon=()=>{setComingSoon(true);window.setTimeout(()=>setComingSoon(false),2200)}; const toggleGroup=(label:string)=>setExpanded(current=>({...current,[label]:!current[label]})); return <><div className={`sidebar-overlay ${open?'show':''}`} onClick={onClose}/><aside className={`sidebar ${open?'open':'closed'}`} aria-label="Primary navigation"><div className="brand"><div className="brand-mark">N</div><div><strong>NIRNAY</strong><small>National Infrastructure Risk<br/> & Nodal Action Intelligence</small></div><button className="icon-button sidebar-close" onClick={onClose} aria-label="Close navigation"><X size={18}/></button></div><nav>{navGroups.map(group=>{const groupHasActive=group.items.some((item)=>{const [,href,key,,status]=item;return status==='active'&&(key==='dashboard'?path==='/':path?.startsWith((href as string).split('?')[0]))});const isExpanded=group.collapsible?(expanded[group.label]??groupHasActive):true;return <div className={`nav-group ${isExpanded?'nav-group-expanded':''}`} key={`${group.label}-${group.items[0][2]}`}><button className={`nav-label ${group.collapsible?'nav-label-collapsible':''}`} type="button" onClick={()=>group.collapsible&&toggleGroup(group.label)} aria-expanded={group.collapsible?isExpanded:undefined} disabled={!group.collapsible}><span>{group.label}</span>{group.collapsible&&<ChevronDown className="nav-chevron" size={14}/>}</button><div className="nav-group-items" aria-hidden={group.collapsible&&!isExpanded}>{isExpanded&&group.items.map((item)=>{const [label,href,key,IconComp,status]=item;const Icon = IconComp as React.ElementType; const enabled=status==='active'||key==='dashboard';const active=enabled&&(key==='dashboard'?path==='/':path?.startsWith((href as string).split('?')[0]));return enabled?<Link className={`nav-item ${active?'active':''}`} href={href as string} key={key as string} onClick={onClose}><Icon size={17}/><span>{label as string}</span></Link>:<button className="nav-item nav-item-disabled" type="button" key={key as string} onClick={showComingSoon} aria-label={`${label}, Coming Soon`}><Icon size={17}/><span>{label as string}</span><small>Coming Soon</small></button>})}</div></div>})}</nav>{comingSoon&&<div className="coming-soon-notice" role="status">Coming Soon</div>}<div className="sidebar-foot"><div className="engine"><span className="status-dot"/>Risk Engine v1.0</div><small>ML Predictions loaded via Provider</small></div></aside></> }
function Header({onMenu,onToggleTheme,dark}:{onMenu:()=>void;onToggleTheme:()=>void;dark:boolean}) { const [notify,setNotify]=useState(false);return <header className="topbar"><div className="top-left"><button className="icon-button menu-toggle" onClick={onMenu} aria-label="Open navigation"><Menu size={20}/></button><div className="top-title">NIRNAY <span>/ Infrastructure Monitoring</span></div></div><div className="top-actions"><span className="updated">Last updated: 31 July 2026, 18:00 IST</span><button className="icon-button" onClick={()=>setNotify(!notify)} aria-label="Notifications"><Bell size={18}/><i className="notification-dot"/></button><button className="icon-button" onClick={onToggleTheme} aria-label={dark?'Switch to light mode':'Switch to dark mode'} title={dark?'Light mode':'Dark mode'}>{dark?<Sun size={18}/>:<Moon size={18}/>}</button><button className="icon-button" aria-label="Help"><CircleHelp size={18}/></button><button className="profile" aria-label="Open user menu"><span>AK</span><div><b>Admin User</b><small>Central Monitoring Unit</small></div><ChevronDown size={15}/></button></div>{notify&&<div className="notification-pop"><b>Notifications</b><p>3 projects require review</p><p>Dataset refreshed successfully</p></div>}</header> }
function Notice(){return <div className="demo-notice"><span>Demonstration Platform</span><p>Using model-generated predictions from the NIRNAY training pipeline. Data is intended to support administrative review.</p></div>}
function Disclaimer(){return <div className="disclaimer-note" style={{fontSize:'0.75rem',color:'var(--text-muted)',marginTop:'12px',borderTop:'1px solid var(--border)',paddingTop:'8px'}}>Predictions are intended to support administrative review and not replace official decision-making.</div>}
function FilterBar({onFilter, allRows, rows}:{onFilter:(m:string,s:string,r:string,st:string)=>void, allRows:Project[], rows:Project[]}) {
  const [m,setM]=useState('All Ministries');
  const [s,setS]=useState('All Sectors');
  const [r,setR]=useState('All Risk Levels');
  const [st,setSt]=useState('Active / Ongoing');
  
  const change=(a:string,b:string,c:string,d:string)=>{
    setM(a);setS(b);setR(c);setSt(d);
    onFilter(a,b,c,d);
  };
  
  const mList = getUniqueMinistries(allRows);
  const sList = getUniqueSectors(allRows);
  const stList = getUniqueStatuses(allRows);
  
  return <div className="filterbar"><div className="filter-label"><Filter size={15}/> Filters</div><select value={m} onChange={e=>change(e.target.value,s,r,st)} aria-label="Ministry"><option>All Ministries</option>{mList.map(x=><option key={x}>{x}</option>)}</select><select value={s} onChange={e=>change(m,e.target.value,r,st)} aria-label="Sector"><option>All Sectors</option>{sList.map(x=><option key={x}>{x}</option>)}</select><select value={r} onChange={e=>change(m,s,e.target.value,st)} aria-label="Risk level"><option>All Risk Levels</option>{['Critical','High','Medium','Low'].map(x=><option key={x}>{x}</option>)}</select><select value={st} onChange={e=>change(m,s,r,e.target.value)} aria-label="Project status"><option>All Statuses</option>{stList.map(x=><option key={x}>{x}</option>)}</select><button className="button ghost" onClick={()=>change('All Ministries','All Sectors','All Risk Levels','Active / Ongoing')}>Reset</button><ExportButton rows={rows}/></div> 
}
function Kpis({rows, counts}:{rows:Project[], counts: Record<string, number>}){const critical=rows.filter(p=>p.riskLevel==='Critical').length,high=rows.filter(p=>p.riskLevel==='High').length,medium=rows.filter(p=>p.riskLevel==='Medium').length,low=rows.filter(p=>p.riskLevel==='Low').length;const exposure=formatCr(rows.reduce((a,p)=>a+Math.max(0,p.revisedCost-p.originalCost),0));const avgDelay = rows.length ? Math.round(rows.reduce((a,p)=>a+p.delayRisk,0)/rows.length) : 0;return <div className="kpi-grid">{[['Total Projects',rows.length||0,'Portfolio'],['Critical Risk',critical,'Requires immediate review'],['High Risk',high,'Priority monitoring'],['Avg Delay Risk',avgDelay+'%','Portfolio average'],['Cost Risk Exposure',exposure,'Model estimate'],['Early Warnings',critical,'Active signals']].map(([label,value,meta],i)=><div className="kpi" key={label}><div className="kpi-top"><span>{label}</span><span className={`kpi-icon k${i}`} /></div><strong>{value}</strong><small>{meta}</small></div>)}</div>}
function ProjectTable({rows,title='Projects Requiring Attention'}:{rows:Project[];title?:string}){return <Card title={title} subtitle="Projects with the highest predicted implementation risk"><div className="table-wrap"><table><caption className="sr-only">{title}</caption><thead><tr><th>Project</th><th>Ministry</th><th>Sector</th><th>Risk</th><th>Cost Risk</th><th>Delay Risk</th><th>Primary Driver</th><th>Last Updated</th><th>Action</th></tr></thead><tbody>{rows.length?rows.map(p=><tr key={p.id}><td><Link className="project-link" href={`/projects/${p.id}`}>{p.name}</Link><small>{p.id}</small></td><td>{p.ministry}</td><td>{p.sector}</td><td><RiskBadge level={p.riskLevel} score={p.riskScore}/></td><td>{p.costRisk}%</td><td>{p.delayRisk}%</td><td>{p.primaryDriver}</td><td>31 Jul 2026</td><td><Link className="view-link" href={`/projects/${p.id}`}>View <ChevronRight size={14}/></Link></td></tr>):<tr><td colSpan={9}><div className="empty">No projects found<br/><button className="text-button">Clear filters</button></div></td></tr>}</tbody></table></div></Card>}
function Distribution({counts}:{counts: Record<string, number>}){const total=Object.values(counts).reduce((a,b)=>a+b,0);return <Card title="Project Risk Distribution" subtitle="ML Predicted portfolio risk classification"><div className="distribution"><div className="distribution-bar">{Object.entries(counts).map(([key,val])=><div key={key} className={`segment ${riskClass(key as RiskLevel)}`} style={{width:`${total===0?0:val/total*100}%`}} title={`${key}: ${val}`}/>)}</div>{Object.entries(counts).map(([key,val])=><div className="legend-row" key={key}><span className={`legend-dot ${riskClass(key as RiskLevel)}`}/><b>{key}</b><span>{val.toLocaleString('en-IN')}</span><small>{total===0?0:Math.round(val/total*100)}%</small></div>)}</div></Card>}
function Trend(){const [range,setRange]=useState(12);const values=range===6?[48,51,54,58,62,66]:range===24?[41,43,46,49,52,55,58,61,64,67,70,73]:[44,46,48,51,54,57,59,61,64,66,69,72];const points=values.map((v,i)=>`${i*(100/(values.length-1))},${100-v}`).join(' ');return <Card title="Portfolio Risk Trend" subtitle="Average predicted risk score across reporting periods"><div className="range-tabs">{[6,12,24].map(n=><button className={range===n?'selected':''} onClick={()=>setRange(n)} key={n}>{n} Months</button>)}</div><div className="trend-chart"><div className="y-labels"><span>100</span><span>75</span><span>50</span><span>25</span><span>0</span></div><svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Portfolio risk trend rising over selected months"><line x1="0" y1="25" x2="100" y2="25"/><line x1="0" y1="50" x2="100" y2="50"/><line x1="0" y1="75" x2="100" y2="75"/><polyline points={points}/>{values.map((v,i)=><circle key={i} cx={i*(100/(values.length-1))} cy={100-v} r="1.4"/>)}</svg></div><div className="x-labels"><span>Jan</span><span>Mar</span><span>May</span><span>Jul</span></div></Card>}
function HomeHero(){return <section className="home-hero"><div className="hero-copy"><span className="hero-kicker">NATIONAL INFRASTRUCTURE INTELLIGENCE PLATFORM</span><h2>See risk earlier.<br/><em>Act with confidence.</em></h2><p>NIRNAY brings project signals, predictive risk intelligence, and coordinated interventions into one clear command surface for infrastructure leaders.</p><div className="hero-actions"><Link href="/projects" className="button hero-primary">Explore project portfolio <ChevronRight size={16}/></Link><Link href="/intelligence" className="hero-text-link">Ask NIRNAY a question <ChevronRight size={15}/></Link></div></div><div className="hero-signal"><div className="signal-top"><span>PORTFOLIO SIGNAL</span><b>LIVE PREDICTIONS</b></div><div className="signal-score"><strong>ML</strong><span>predictions active</span></div><div className="signal-bars"><div><span>Cost exposure</span><Bar value={72} color="amber"/></div><div><span>Schedule pressure</span><Bar value={58} color="blue"/></div><div><span>Intervention readiness</span><Bar value={86} color="green"/></div></div><div className="signal-foot"><span className="status-dot"/> Risk Engine v1.0 <span>·</span> updated 31 Jul 2026</div></div></section>}
function Landing({dark,onToggleTheme}:{dark:boolean;onToggleTheme:()=>void}){return <div className={`landing-page ${dark?'dark-mode':''}`}><header className="landing-nav"><Link href="/" className="landing-brand"><span className="brand-mark">N</span><span><b>NIRNAY</b><small>National Infrastructure Risk & Nodal Action Intelligence</small></span></Link><div className="landing-actions"><a href="#capabilities">Capabilities</a><a href="#approach">Our approach</a><button className="landing-theme" onClick={onToggleTheme} aria-label={dark?'Switch to light mode':'Switch to dark mode'}>{dark?<Sun size={17}/>:<Moon size={17}/>}</button><Link href="/dashboard" className="button landing-login">Open dashboard <ChevronRight size={15}/></Link></div></header><main><section className="landing-hero"><div className="landing-hero-copy"><span className="hero-kicker">NATIONAL INFRASTRUCTURE INTELLIGENCE PLATFORM</span><h1>Make every infrastructure decision <em>more certain.</em></h1><p>NIRNAY gives public institutions a clear, evidence-led view of project risk — so leaders can see emerging issues early, coordinate action, and deliver with confidence.</p><div className="landing-cta"><Link href="/dashboard" className="button hero-primary">Enter monitoring dashboard <ChevronRight size={16}/></Link><span><ShieldCheck size={16}/> Built for accountable public delivery</span></div></div><div className="landing-visual"><div className="visual-heading"><span>PORTFOLIO INTELLIGENCE</span><b>ML POWERED</b></div><div className="visual-number">Live<span>predictions</span></div><div className="visual-bars"><div><span><b>Cost exposure</b><b>72%</b></span><Bar value={72} color="amber"/></div><div><span><b>Schedule pressure</b><b>58%</b></span><Bar value={58} color="blue"/></div><div><span><b>Intervention readiness</b><b>86%</b></span><Bar value={86} color="green"/></div></div><div className="visual-foot"><span className="status-dot"/> Risk Engine v1.0 <span>Updated 31 Jul 2026</span></div></div></section><section id="capabilities" className="landing-section"><div className="section-intro"><span className="hero-kicker">ONE COMMAND SURFACE</span><h2>From project signals to timely action.</h2><p>Connect the view across your portfolio with practical intelligence designed for monitoring units, programme directors, and senior decision-makers.</p></div><div className="capability-grid"><article><ShieldCheck size={20}/><h3>Predictive risk intelligence</h3><p>Surface cost, schedule, and delivery risks before they become irreversible.</p></article><article><Zap size={20}/><h3>Early warning signals</h3><p>Turn fragmented reporting into clear signals that focus attention where it matters.</p></article><article><SlidersHorizontal size={20}/><h3>Coordinated interventions</h3><p>Move from diagnosis to accountable action with a shared operational picture.</p></article></div></section><section id="approach" className="landing-proof"><div><span className="hero-kicker">DESIGNED FOR TRUST</span><h2>Evidence for better public delivery.</h2></div><div className="proof-items"><span><b>ML Powered</b><small>predictions</small></span><span><b>24/7</b><small>portfolio visibility</small></span><span><b>100%</b><small>decision traceability</small></span></div></section></main><footer className="landing-footer"><span>NIRNAY</span><small>National Infrastructure Risk & Nodal Action Intelligence</small><Link href="/dashboard">Continue to dashboard <ChevronRight size={14}/></Link></footer></div>}

function Dashboard(){
  const [allRows, setAllRows] = useState<Project[]>([]);
  const [rows,setRows]=useState<Project[]>([]);

  useEffect(() => {
    getEnrichedProjects().then(r => { setAllRows(r); setRows(r); });
  }, []);

  const counts = {
    Critical: rows.filter(p => p.riskLevel === 'Critical').length,
    High: rows.filter(p => p.riskLevel === 'High').length,
    Medium: rows.filter(p => p.riskLevel === 'Medium').length,
    Low: rows.filter(p => p.riskLevel === 'Low').length
  };
  
  const warnings = rows.filter(p => p.riskLevel === 'Critical' && p.delayRisk > 75).slice(0, 8).map(p => ({
    title: `${p.primaryDriver} detected`,
    project: p.name,
    detail: `Model flags a ${p.delayRisk}% probability of schedule delay driven by ${p.primaryDriver.toLowerCase()}.`,
    metric: `Probability: ${p.delayRisk}%`,
    id: p.id,
    severity: p.riskLevel
  }));

  const handleFilter = (m:string,s:string,r:string,st:string) => {
    const filtered = filterProjects(allRows, '', m, s, r, st);
    setRows(filtered);
  };

  return <><HomeHero/><PageHeader title="Infrastructure Project Monitoring" subtitle="Predictive monitoring and early intervention across the infrastructure portfolio"/><Notice/><FilterBar allRows={allRows} rows={rows} onFilter={handleFilter}/><Kpis rows={rows} counts={counts}/><ProjectTable rows={rows.slice().sort((a,b)=>b.riskScore-a.riskScore).slice(0,5)}/><div className="two-col"><Distribution counts={counts}/><Trend/></div><Card title="Early Warning Signals" subtitle="ML signals requiring administrative review"><div className="warnings">{warnings.length?warnings.map(w=><div className="warning-row" key={w.id}><div className="warning-icon"><Zap size={17}/></div><div><b>{w.title}</b><strong>{w.project}</strong><p>{w.detail}</p><small>{w.metric}</small></div><Link href={`/projects/${w.id}`} className="button secondary">View Project</Link></div>):<div className="empty">No warnings for current selection</div>}</div></Card></>
}

function ProjectsPage(){
  const [query,setQuery]=useState('');
  const [risk,setRisk]=useState('All Risk Levels');
  const [allRows, setAllRows] = useState<Project[]>([]);
  useEffect(() => { getEnrichedProjects().then(setAllRows); }, []);

  const rows = filterProjects(allRows, query, 'All Ministries', 'All Sectors', risk, 'Active / Ongoing');
  
  return <><PageHeader title="All Projects" subtitle="Explore and review the national infrastructure project portfolio"/><Notice/><div className="explorer-tools"><div className="searchbox"><Search size={16}/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search projects..." aria-label="Search projects"/></div><select value={risk} onChange={e=>setRisk(e.target.value)} aria-label="Risk filter"><option>All Risk Levels</option><option>Critical</option><option>High</option><option>Medium</option><option>Low</option></select><button className="button secondary"><Filter size={15}/> More filters</button><ExportButton rows={rows}/></div><Card title="Project Portfolio" subtitle={`${rows.length} records in current view`}><div className="table-wrap"><table><thead><tr><th>Project</th><th>Ministry</th><th>Sector</th><th>Original Cost</th><th>Expenditure</th><th>Physical Progress</th><th>Risk</th><th>Status</th></tr></thead><tbody>{rows.length?rows.map(p=><tr key={p.id}><td><Link className="project-link" href={`/projects/${p.id}`}>{p.name}</Link><small>{p.id}</small></td><td>{p.ministry}</td><td>{p.sector}</td><td>{formatCr(p.originalCost)}</td><td>{formatCr(p.expenditure)}</td><td>{p.physicalProgress==null?'N/A':`${Number(p.physicalProgress).toFixed(1)}%`}</td><td>{p.lifecycle?.includes('Completed') ? '-' : <RiskBadge level={p.riskLevel} score={p.riskScore}/>}</td><td><span className="status-badge">{p.lifecycle||p.status}</span></td></tr>):<tr><td colSpan={8}><div className="empty">No projects found for current filter</div></td></tr>}</tbody></table></div></Card></>
}

function Detail({id}:{id:string}){
  const [p, setP] = useState<Project | null>(null);
  const [ml, setML] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [tab, setTab] = useState('Overview');
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    getEnrichedProjects().then(projects => {
      const match = projects.find(pr => pr.id === String(id) || pr.id.replace(/^0+/,'') === String(id).replace(/^0+/,''));
      setP(match || null);
      setLoaded(true);
    });
    getMLPredictionByProjectId(id).then(setML);
    getProjectHistoryData(id).then(setHistory);
  }, [id]);

  if (!loaded) return <div style={{padding:'40px',textAlign:'center'}}>Loading...</div>;
  if (!p) return <div className="empty" style={{marginTop:'100px'}}><h2>Project not found</h2><p>The requested project ID ({id}) does not exist in the dataset.</p></div>;

  const renderTabContent = () => {
    if (tab === 'Risk Analysis') return <Card title="SHAP Feature Explanations" subtitle="Top risk drivers and their relative impact on the delay prediction">
      {ml?.explanation ? <div className="table-wrap"><table><thead><tr><th>Feature</th><th>Impact</th></tr></thead><tbody>{Object.entries(ml.explanation).slice(0,10).map(([k,v]:any)=><tr key={k}><td>{k}</td><td><Bar value={Math.min(100, Math.abs(v)*20)} color={v>0?'red':'green'}/></td></tr>)}</tbody></table></div> : <div className="empty">No explanation available</div>}
    </Card>;
    if (tab === 'History') return <Card title="Historical Data" subtitle="Monthly records"><div className="table-wrap"><table><thead><tr><th>Month</th><th>Physical</th><th>Financial</th><th>Risk Score</th></tr></thead><tbody>{history.map((h:any,i)=>{const scoreVal = typeof h.score==='number'?h.score:(typeof h.ep==='number'&&h.ep>0?Math.round(h.ep):(h.r==='Critical'?85:h.r==='High'?70:h.r==='Medium'?45:20));return <tr key={i}><td>{h.d}</td><td>{h.p}%</td><td>{h.financialProgress||0}%</td><td><RiskBadge level={h.r||'Medium'} score={scoreVal}/></td></tr>;})}</tbody></table></div></Card>;
    if (tab === 'Interventions') return <Card title="Recommended Interventions" subtitle="Rule-based recommendations"><div className="warnings"><div className="warning-row"><div className="warning-icon"><SlidersHorizontal size={17}/></div><div><b>Review Project Timeline</b><strong>{p.name}</strong><p>Delay risk is {p.delayRisk}%. Immediate review required based on primary driver: {p.primaryDriver}.</p><small>Priority: {p.riskLevel}</small></div></div></div></Card>;
    return null;
  }

  const isCompleted = p.lifecycle === 'Completed - Delayed' || p.lifecycle === 'Completed on Schedule';
  const physProg = p.physicalProgress == null ? 'Not available' : `${Number(p.physicalProgress).toFixed(1)}%`;
  const finProg = p.financialProgress == null ? 'Not available' : `${Number(p.financialProgress).toFixed(1)}%`;
  
  return <><PageHeader title={p.name} subtitle={`${p.id} · ${p.ministry} · ${p.sector} · ${p.state}`}/><div className="detail-hero"><div><span className="eyebrow">{isCompleted ? 'COMPLETED PROJECT RECORD' : 'PROJECT RECORD'}</span><h2>{p.name}</h2><p>Implementing agency: {p.implementingAgency}</p></div>{isCompleted ? <span className="status-badge" style={{fontSize:'16px', padding:'10px'}}>{p.lifecycle}</span> : <RiskBadge level={p.riskLevel} score={p.riskScore}/>}</div><div className="tabs">{['Overview','Financial','Schedule','Milestones','Risk Analysis','Interventions','History'].map(t=><button key={t} className={tab===t?'selected':''} onClick={()=>setTab(t)}>{t}</button>)}</div><div className="detail-grid">{[['Approved cost',formatCr(p.originalCost)],['Revised cost',formatCr(p.revisedCost)],['Expenditure',formatCr(p.expenditure)],['Start date','Not available'],['Planned completion',p.plannedCompletion||'Not available'],['Expected completion',p.expectedCompletion||'Not available'],['Physical progress',physProg],['Financial progress',finProg],['Implementing agency',p.implementingAgency||'Not available'],['State / region',p.state||'Not available'],['Project status',p.lifecycle||p.status],['Assessment',isCompleted?'Historical Outcome':'Model-based estimate']].map(([l,v])=><div className="detail-item" key={l}><small>{l}</small><b>{v}</b></div>)}</div><div className="two-col detail-sections">
  {isCompleted ? <Card title="POST-PROJECT OUTCOME" subtitle="Final completion metrics"><div className="risk-metrics"><div><b style={{fontSize:'1.8rem', color:'var(--text)'}}>{p.timeOverrunMonths||0}</b><span>Months Delayed</span></div><div><b>{formatCr(p.revisedCost-p.originalCost)}</b><span>Cost Overrun</span></div></div></Card> : <Card title="ML RISK ASSESSMENT" subtitle="Model-based predictive analysis">
    {(p.riskLevel as unknown as string) === 'Prediction unavailable' ? (
      <div className="empty">Prediction unavailable</div>
    ) : (
      <>
        <div className="risk-metrics">
          <div><b style={{fontSize:'1.8rem', color:'var(--text)'}}>{ml?.risk_level_ml?.toUpperCase()}</b><span>Risk Level</span></div>
          <div><b>{Math.round((ml?.risk_confidence||0)*100)}%</b><span>Model Confidence</span></div>
          <div><b>{p.costRisk}%</b><span>Cost Risk Exposure</span></div>
        </div>
        <div style={{marginTop: '20px', marginBottom: '10px'}}>
          <div style={{display:'flex',justifyContent:'space-between',marginBottom:'4px',fontSize:'0.85rem'}}><span>Critical</span><span>{Math.round((ml?.probability_critical||0)*100)}%</span></div><Bar value={(ml?.probability_critical||0)*100} color="red"/>
          <div style={{display:'flex',justifyContent:'space-between',marginBottom:'4px',fontSize:'0.85rem',marginTop:'8px'}}><span>High</span><span>{Math.round((ml?.probability_high||0)*100)}%</span></div><Bar value={(ml?.probability_high||0)*100} color="amber"/>
          <div style={{display:'flex',justifyContent:'space-between',marginBottom:'4px',fontSize:'0.85rem',marginTop:'8px'}}><span>Medium</span><span>{Math.round((ml?.probability_medium||0)*100)}%</span></div><Bar value={(ml?.probability_medium||0)*100} color="blue"/>
          <div style={{display:'flex',justifyContent:'space-between',marginBottom:'4px',fontSize:'0.85rem',marginTop:'8px'}}><span>Low</span><span>{Math.round((ml?.probability_low||0)*100)}%</span></div><Bar value={(ml?.probability_low||0)*100} color="green"/>
        </div>
        <h3 style={{marginTop:'24px', fontSize:'1rem'}}>Top Risk Drivers</h3>
        {ml?.top_risk_driver_1 && <div className="driver"><span>1. {ml.top_risk_driver_1}</span></div>}
        {ml?.top_risk_driver_2 && <div className="driver"><span>2. {ml.top_risk_driver_2}</span></div>}
        {ml?.top_risk_driver_3 && <div className="driver"><span>3. {ml.top_risk_driver_3}</span></div>}
        <Disclaimer/>
      </>
    )}
  </Card>}<Card title="Risk Trajectory" subtitle="Monthly risk history">{(() => {
    const MONTH_ABBR = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    // 1. Sort chronologically by YYYY-MM (ISO string sort = chronological, year preserved)
    const sorted = [...history].sort((a:any, b:any) => String(a.d) < String(b.d) ? -1 : String(a.d) > String(b.d) ? 1 : 0);
    // 2. Deduplicate: one record per YYYY-MM (last record per period wins)
    const seen = new Map<string, any>();
    for (const pt of sorted) { seen.set(String(pt.d), pt); }
    const deduped = Array.from(seen.values());
    // 3. Take last 12 chronological records
    const points = deduped.slice(-12);
    if (!points.length) return <div className="empty">Insufficient historical data</div>;
    // 4. Compute scores and labels
    const items = points.map((pt:any) => {
      // ep is the actual historical risk score (0-100); s is a string status, not a number
      const scoreVal = (typeof pt.ep === 'number' && pt.ep > 0)
        ? Math.round(pt.ep)
        : (pt.r === 'Critical' ? 85 : pt.r === 'High' ? 70 : pt.r === 'Medium' ? 45 : 20);
      const color = scoreVal > 75 ? '#ae5a5b' : scoreVal > 50 ? '#c27a43' : scoreVal > 25 ? '#b08c3c' : '#5a9474';
      // Label: "Apr 2025" from "2025-04"
      const parts = String(pt.d).split('-');
      const year = parts[0] || '';
      const monthIdx = parts[1] ? parseInt(parts[1], 10) - 1 : -1;
      const label = (monthIdx >= 0 && monthIdx < 12) ? `${MONTH_ABBR[monthIdx]} ${year}` : String(pt.d);
      return { scoreVal, color, label, d: String(pt.d) };
    });
    return <><div className="trajectory">{items.map((item, i) => (
      <div key={i}>
        <span>{item.scoreVal}%</span>
        <div style={{height:`${item.scoreVal}%`, background: item.color}} title={`${item.d} — Risk: ${item.scoreVal}%`}/>
        <small>{item.label}</small>
      </div>
    ))}</div></>;
  })()}</Card>{renderTabContent()}</div></>
}

function GenericPage({title,subtitle}:{title:string;subtitle:string}){
  const [allRows, setAllRows] = useState<Project[]>([]);
  const [rows, setRows] = useState<Project[]>([]);
  useEffect(() => { getEnrichedProjects().then(r => { setAllRows(r); setRows(r); }); }, []);
  const handleFilter = (m:string,s:string,r:string,st:string) => setRows(filterProjects(allRows, '', m, s, r, st));
  
  if (title === 'Intervention Center') return <><PageHeader title={title} subtitle={subtitle}/><Notice/><FilterBar allRows={allRows} rows={rows} onFilter={handleFilter}/><Card title="Interventions" subtitle="Recommended Actions"><div className="table-wrap"><table><thead><tr><th>Project</th><th>Risk Driver</th><th>Recommended Action</th><th>Priority</th><th>Target Date</th></tr></thead><tbody>{rows.filter(x=>x.riskLevel==='Critical' || x.riskLevel==='High').slice(0,30).map(p=><tr key={p.id}><td><Link href={`/projects/${p.id}`} className="project-link">{p.name}</Link><small>{p.id}</small></td><td>{p.primaryDriver}</td><td>{p.primaryDriver.includes('progress')?'Progress escalation':p.primaryDriver.includes('cost')?'Financial review':'Schedule recovery review'}</td><td><RiskBadge level={p.riskLevel} score={p.riskScore}/></td><td><span className="status-badge">Next 30 days</span></td></tr>)}</tbody></table></div></Card></>;
  
  if (title === 'Model Performance') return <><PageHeader title={title} subtitle={subtitle}/><Notice/><FilterBar allRows={allRows} rows={rows} onFilter={handleFilter}/><Card title="Model Metrics" subtitle="HistGradientBoosting (scikit-learn)"><div className="table-wrap"><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody><tr><td>Accuracy</td><td>94.79%</td></tr><tr><td>Precision</td><td>96.45%</td></tr><tr><td>Recall</td><td>95.07%</td></tr><tr><td>F1 Score</td><td>95.76%</td></tr><tr><td>ROC-AUC</td><td>0.979</td></tr></tbody></table></div></Card></>;
  
  // Default Analytics
  const avgRisk = rows.length ? Math.round(rows.reduce((a,b)=>a+b.riskScore,0)/rows.length) : 0;
  const avgDelay = rows.length ? Math.round(rows.reduce((a,b)=>a+b.delayRisk,0)/rows.length) : 0;
  
  // Sector analysis
  const sectorGroups = rows.reduce((acc, p) => { if(!acc[p.sector]) acc[p.sector] = []; acc[p.sector].push(p); return acc; }, {} as any);
  const sectorData = Object.keys(sectorGroups).map(k => ({sector: k, count: sectorGroups[k].length, risk: Math.round(sectorGroups[k].reduce((a:any,b:any)=>a+b.riskScore,0)/sectorGroups[k].length)})).sort((a,b)=>b.risk-a.risk).slice(0,10);

  return <><PageHeader title={title} subtitle={subtitle}/><Notice/><FilterBar allRows={allRows} rows={rows} onFilter={handleFilter}/><div className="analytics-grid"><Card title="Total Projects" subtitle="Monitored portfolio"><div className="large-stat">{rows.length.toLocaleString()}</div><p className="muted">Across {Object.keys(sectorGroups).length} sectors</p></Card><Card title="Average Risk Score" subtitle="Portfolio baseline"><div className="large-stat">{avgRisk} / 100</div><p className="muted">Based on ML inference</p></Card><Card title="Average Delay Risk" subtitle="Schedule indicator"><div className="large-stat">{avgDelay}%</div><p className="muted">Probability &gt; 6 months</p></Card><Card title="Total Exposure" subtitle="Cost variance"><div className="large-stat">{formatCr(rows.reduce((a,b)=>a+Math.max(0,b.revisedCost-b.originalCost),0))}</div><p className="muted">Estimated variance</p></Card></div>
  <Card title="Sector Analysis" subtitle="Highest risk infrastructure sectors"><div className="table-wrap"><table><thead><tr><th>Sector</th><th>Projects</th><th>Average Risk Score</th></tr></thead><tbody>{sectorData.map(s=><tr key={s.sector}><td>{s.sector}</td><td>{s.count}</td><td><RiskBadge level={s.risk>75?'Critical':s.risk>50?'High':s.risk>25?'Medium':'Low'} score={s.risk}/></td></tr>)}</tbody></table></div></Card></>
}

function Intelligence(){
  const [answer,setAnswer]=useState('');
  const [allRows, setAllRows] = useState<Project[]>([]);
  const [rows, setRows] = useState<Project[]>([]);
  useEffect(() => { getEnrichedProjects().then(projects => { setAllRows(projects); setRows(projects); }); }, []);
  const handleFilter = (m:string,s:string,r:string,st:string) => setRows(filterProjects(allRows, '', m, s, r, st));
  return <><PageHeader title="Project Intelligence" subtitle="Query project monitoring data using natural language."/><Notice/><FilterBar allRows={allRows} rows={rows} onFilter={handleFilter}/><Card title="Analytical query" subtitle="Select a question to review structured portfolio results"><div className="querybox"><Search size={18}/><input placeholder="Ask about projects, risks, delays or cost trends..." value={answer} onChange={e=>setAnswer(e.target.value)}/><button className="button primary" onClick={()=>setAnswer(answer||'Projects requiring immediate intervention')}>Analyze</button></div><div className="query-chips">{['Projects requiring immediate intervention','Projects with increasing risk','Highest schedule risk sectors','Why is this project high risk?','Projects with similar historical patterns'].map(q=><button key={q} onClick={()=>setAnswer(q)}>{q}</button>)}</div>{answer&&<div className="intelligence-answer"><span className="eyebrow">STRUCTURED ANALYTICAL RESPONSE</span><h3>Query executed against current prediction payload.</h3><p>Results are based on the latest available ML inferences.</p><ProjectTable title="Matching projects" rows={rows.filter(p => { const q = answer.toLowerCase(); if(q.includes('rail') && !p.ministry.toLowerCase().includes('rail')) return false; if((q.includes('high') || q.includes('increasing')) && p.riskLevel !== 'High') return false; if((q.includes('critical') || q.includes('intervention')) && p.riskLevel !== 'Critical') return false; return true; }).slice(0,5)}/></div>}</Card></>
}

function Simulation(){
  const [rows, setRows] = useState<Project[]>([]);
  const [pId, setPId] = useState<string>('');
  const [progress,setProgress]=useState(0);
  const [exp,setExp]=useState(0);
  
  useEffect(() => { 
    getEnrichedProjects().then(r => { 
      setRows(r);
      const crit = r.find(x => x.riskLevel === 'Critical');
      if (crit) { setPId(crit.id); setProgress(crit.physicalProgress); setExp(Math.round((crit.expenditure/Math.max(1,crit.revisedCost))*100)); }
    }); 
  }, []);

  const p = rows.find(x => x.id === pId);
  const riskChange = p ? Math.round((p.physicalProgress - progress) * 0.5 + (exp - (p.expenditure/Math.max(1,p.revisedCost)*100)) * -0.5) : 0;
  const newDelay = p ? Math.max(0, Math.min(100, p.delayRisk + riskChange)) : 0;
  const newRiskLvl = newDelay > 75 ? 'Critical' : newDelay > 50 ? 'High' : newDelay > 25 ? 'Medium' : 'Low';

  return <><PageHeader title="Project Scenario Analysis" subtitle="Test planning assumptions and review model-based outcomes."/><Notice/>
  {rows.length > 0 && <div style={{marginBottom:'20px'}}>Select Project: <select value={pId} onChange={e=>{setPId(e.target.value); const np=rows.find(x=>x.id===e.target.value); if(np){setProgress(np.physicalProgress); setExp(Math.round((np.expenditure/Math.max(1,np.revisedCost))*100));}}} style={{padding:'5px', marginLeft:'10px', width:'400px'}}>{rows.filter(x=>x.riskLevel==='Critical'||x.riskLevel==='High').slice(0,50).map(x=><option key={x.id} value={x.id}>{x.name}</option>)}</select></div>}
  <div className="simulation-grid"><Card title="Scenario inputs" subtitle="Adjust assumptions for the prototype model"><label>Physical progress <b>{progress}%</b><input type="range" min="0" max="100" value={progress} onChange={e=>setProgress(+e.target.value)}/></label><label>Financial progress <b>{exp}%</b><input type="range" min="0" max="100" value={exp} onChange={e=>setExp(+e.target.value)}/></label><label>Milestone completion <b>{Math.min(100,progress+8)}%</b><input type="range" min="20" max="100" defaultValue={progress}/></label><label>Completion date <input type="date" defaultValue="2027-01-31"/></label></Card><Card title="Scenario outcome" subtitle="Estimated approximation (Not a fresh ML inference)"><div className="outcome-grid"><div><span>Current Risk</span><b style={{fontSize:'20px'}}>{p?.riskLevel || '-'}</b></div><div className="emphasis"><span>Scenario Risk</span><b style={{fontSize:'20px'}}>{newRiskLvl}</b></div><div><span>Current Delay</span><b style={{fontSize:'20px'}}>{p?.delayRisk || 0}%</b></div><div className="emphasis"><span>Scenario Delay</span><b style={{fontSize:'20px'}}>{newDelay}%</b></div></div><div className="scenario-note">Simulation calculates deterministic estimates based on current model thresholds. It does not replace the official ML inference pipeline.</div></Card></div></>
}

export default function NirnayApp(){
  const path=usePathname();
  const [menu,setMenu]=useState(false);
  const [dark,setDark]=useState(false);
  const [themeReady,setThemeReady]=useState(false);
  
  useEffect(()=>{
    const saved=window.localStorage.getItem('nirnay-theme');
    setDark(saved ? saved==='dark' : window.matchMedia('(prefers-color-scheme: dark)').matches);
    setThemeReady(true);
  },[]);
  
  useEffect(()=>{
    if(!themeReady) return;
    window.localStorage.setItem('nirnay-theme',dark?'dark':'light');
    document.documentElement.classList.toggle('dark-mode',dark);
  },[dark,themeReady]);
  
  let content:React.ReactNode;
  if(path?.startsWith('/projects/'))content=<Detail id={path.split('/')[2]}/>;
  else if(path==='/projects')content=<ProjectsPage/>;
  else if(path==='/intelligence')content=<Intelligence/>;
  else if(path==='/simulation')content=<Simulation/>;
  else if(path==='/dashboard')content=<Dashboard/>;
  else if(path==='/')return <Landing dark={dark} onToggleTheme={()=>setDark(v=>!v)}/>;
  else {
    const title=path?.includes('data-quality')?'Data Quality':path?.includes('model-performance')?'Model Performance':path?.includes('interventions')?'Intervention Center':path?.includes('warnings')?'Early Warning Signals':path?.includes('analytics')?'Analytics Overview':'Administration';
    content=<GenericPage title={title} subtitle="Portfolio indicators and operational records for administrative review."/>
  }
  return <div className={`app-shell ${dark?'dark-mode':''}`}><Sidebar open={menu} onClose={()=>setMenu(false)}/><div className="main-area"><Header onMenu={()=>setMenu(true)} onToggleTheme={()=>setDark(v=>!v)} dark={dark}/><main className="content">{content}</main><footer><b>NIRNAY</b><span>National Infrastructure Risk & Nodal Action Intelligence</span><small>Demonstration prototype — project data is derived from supplied PAIMANA reports and model-generated predictions; it is not an official government record. Predictive outputs support, but do not replace, administrative judgement.</small></footer></div></div>
}
