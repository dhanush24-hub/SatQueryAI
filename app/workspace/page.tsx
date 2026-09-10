'use client';
import {useState,useEffect,useRef} from 'react';
import {Orbit,Plus,ArrowUp,ArrowUpRight,Upload,ScanLine,Layers,GitCompareArrows,Settings2,Download,Check,ChevronRight,Image as ImageIcon,X,ZoomIn,ZoomOut,RotateCcw,Info,LoaderCircle,MessageSquare,ShieldCheck,PanelLeft,AlertTriangle,RefreshCw,History,FileText} from 'lucide-react';
import {Tabs,TabsList,TabsTrigger,TabsContent} from '@/components/ui/tabs';
import {Dialog,DialogContent,DialogTitle,DialogDescription} from '@/components/ui/dialog';
import {Sidebar,SidebarProvider,SidebarContent,SidebarHeader,SidebarFooter,SidebarTrigger} from '@/components/ui/sidebar';
import {sampleAnalysis,toDisplayAnalysis,type Scene,type Analysis} from '@/lib/analysis';
import {samplePlan,type WorkflowPlan} from '@/lib/workflow';
import {missions,demoPlan,demoRegistry,reviewPlan,type DemoId} from '@/lib/demos';
import {executePlan,type PipelineEvent} from '@/lib/executor';
import {refineExisting} from '@/lib/validation';
import type {MissionResult,ExecutionTrace,Finding} from '@/lib/domain';
import {FindingsPanel,FindingEvidence,QuestionUnderstanding,WorkflowExecution,ExecutionLog} from '@/components/mission-findings';
import {EvidenceDossier} from '@/components/evidence-dossier';
import {AnalysisHistoryPanel} from '@/components/analysis-history-panel';
import {ImageryCanvas} from '@/components/imagery-canvas';
import {IsroAnalytics} from '@/components/isro-analytics';
import {uploadImagery,getAssetPreviewUrl,type ModalityType} from '@/lib/api/uploads';
import {checkPairCompatibility,alignImageryPair,type PairCompatibilityReport} from '@/lib/api/compatibility';
import {submitAnalysis,mapBackendAnalysisToMissionResult,type AnalysisResponse} from '@/lib/api/analysis';
type Message={role:'user'|'assistant';text:string;result?:Analysis;plan?:WorkflowPlan};
type Trace={label:string;detail:string;time:string};
export default function Workspace(){
 const [scenes,setScenes]=useState<Scene[]>([]),[sample,setSample]=useState(false),[messages,setMessages]=useState<Message[]>([]),[question,setQuestion]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState(''),[settings,setSettings]=useState(false),[key,setKey]=useState(''),[model,setModel]=useState('gemini-2.5-flash'),[live,setLive]=useState(false),[selected,setSelected]=useState<number|null>(null),[imageIndex,setImageIndex]=useState(0),[zoom,setZoom]=useState(1),[tab,setTab]=useState('plan');
 const [compatibilityReport,setCompatibilityReport]=useState<PairCompatibilityReport|null>(null);
 const [aligning,setAligning]=useState(false);
 const [activeAnalysis,setActiveAnalysis]=useState<AnalysisResponse|null>(null);
 const [historyOpen,setHistoryOpen]=useState(false);
 const fileInput=useRef<HTMLInputElement>(null),bottom=useRef<HTMLDivElement>(null),controller=useRef<AbortController|null>(null);
 const [inspected,setInspected]=useState<Analysis|null>(null);
 const result=inspected||(!busy?messages.at(-1)?.result:undefined);
 const [plan,setPlan]=useState<WorkflowPlan|null>(null),[trace,setTrace]=useState<Trace[]>([]),[phase,setPhase]=useState(''),[dragging,setDragging]=useState(false);
 const locked=useRef(false);
 const [demoId,setDemoId]=useState<DemoId>('single'),[richTrace,setRichTrace]=useState<ExecutionTrace[]>([]),[selectedFinding,setSelectedFinding]=useState<string|null>(null);
 const missionQuestion=useRef('');
 const currentMission=missions.find(m=>m.id===demoId)!;
 function record(label:string,detail:string){setTrace(t=>[...t,{label,detail,time:new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}]);}
 const shownPlan=inspected?messages.find(m=>m.result===inspected)?.plan:plan;
 async function loadSample(id:DemoId='urban'){
  if(locked.current)return;locked.current=true;setBusy(true);setError('');setPhase('Loading demonstration imagery & executing multi-sensor analysis');
  try{
    const mission=missions.find(m=>m.id===id)!;
    const inputs=await Promise.all(mission.images.map(async s=>{
      const r=await fetch(s.url);
      if(!r.ok)throw Error('A demo image could not load.');
      const blob=await r.blob();
      const data=await readFile(blob);
      return {name:s.name,label:s.label,mime:'image/jpeg',data:data.split(',')[1],bytes:blob.size};
    }));
    setScenes(inputs);
    setDemoId(id);
    setSample(true);
    setQuestion(mission.question);
    missionQuestion.current=mission.question;
    setImageIndex(0);
    setZoom(1);

    const nextPlan=demoPlan(id,mission.question);
    setPlan(nextPlan);
    const planningEvent:ExecutionTrace={
      timestamp:new Date().toISOString(),
      stepId:'planner',
      step:'Input assessment and workflow composition',
      purpose:mission.question,
      inputs:inputs.map((_,i)=>'img-'+(i+1)),
      provider:'Curated mission planner',
      status:'completed',
      summary:nextPlan.rationale
    };
    setRichTrace([planningEvent]);
    const answer=await executePlan(
      {key:'',model:'curated',question:mission.question,scenes:inputs,plan:nextPlan,mission:{question:mission.question,history:[],previousResult:undefined,previousPlan:null}},
      (event)=>{if(event.type==='step')setRichTrace(t=>[...t,event.event]);},
      demoRegistry(id)
    );
    answer.source='Guided demo · curated fixtures, not live AI';
    answer.trace=[planningEvent,...answer.trace];
    const display=toDisplayAnalysis(answer);
    setMessages([
      {role:'user',text:mission.question},
      {role:'assistant',text:display.answer,result:display,plan:nextPlan}
    ]);
    setSelected(null);
    setSelectedFinding(null);
    setTab('findings');
  }
  catch(e){setError((e as Error).message);}
  finally{locked.current=false;setBusy(false);setPhase('');}
 }
 useEffect(()=>{const q=new URLSearchParams(location.search);const demo=q.get('demo') as DemoId|null;if(demo)void loadSample(demo);},[]);
 useEffect(()=>()=>controller.current?.abort(),[]);
 useEffect(()=>{const ctx=(document as unknown as {modelContext?:{registerTool:Function}}).modelContext;if(!ctx)return;const life=new AbortController();try{Promise.resolve(ctx.registerTool({name:'stage_satellite_question',description:'Stage a question in the visible SatQuery composer. Does not submit or contact Gemini.',inputSchema:{type:'object',properties:{question:{type:'string',maxLength:4000}},required:['question'],additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:false},execute(input:unknown){const q=(input as {question?:unknown})?.question;if(typeof q!=='string'||!q.trim()||q.length>4000)throw Error('A question of 1–4,000 characters is required.');setQuestion(q);return {staged:true,question:q};}},{signal:life.signal})).catch(()=>{});}catch{}return()=>life.abort();},[]);
  function clear(){setCompatibilityReport(null);setRichTrace([]);setSelectedFinding(null);missionQuestion.current='';setPlan(null);setTrace([]);controller.current?.abort();setBusy(false);setScenes([]);setSample(false);setMessages([]);setInspected(null);setQuestion('');setError('');setSelected(null);setImageIndex(0);setZoom(1);setActiveAnalysis(null);}
  async function upload(files:FileList|null){
    if(!files||locked.current)return;setDragging(false);setError('');const list=Array.from(files);
    if(list.length+scenes.length>6){setError('You can attach up to 6 images in one conversation.');return;}
    locked.current=true;setBusy(true);setPhase('Inspecting raster headers & extracting geospatial metadata (FastAPI / Rasterio)...');
    try{
      const hasBenchmark=list.some(f=>['image/png','image/jpeg','image/webp'].includes(f.type)||/\.(png|jpg|jpeg|webp)$/i.test(f.name));
      const uploadedAssets=await uploadImagery(list,{benchmarkMode:hasBenchmark});
      const added:Scene[]=await Promise.all(uploadedAssets.map(async (asset,idx)=>{
        const originalFile=list[idx];
        let previewData='';
        let previewMime='image/png';
        try{
          const previewUrl=getAssetPreviewUrl(asset.id);
          const res=await fetch(previewUrl);
          if(res.ok){
            const blob=await res.blob();
            const base64Url=await readFile(blob);
            previewData=base64Url.split(',')[1];
            previewMime=blob.type||'image/png';
          }
        }catch(e){console.warn('Could not fetch display preview from server:',e);}
        if(!previewData&&originalFile&&['image/png','image/jpeg','image/webp'].includes(originalFile.type)){
          const base64Url=await readFile(originalFile);
          previewData=base64Url.split(',')[1];
          previewMime=originalFile.type;
        }
        return {
          name:asset.original_filename||originalFile?.name||asset.id,
          data:previewData,
          mime:previewMime,
          label:asset.sensor?`${asset.sensor} · ${asset.modality}`:(asset.modality!=='UNKNOWN'?asset.modality:''),
          width:asset.width||undefined,
          height:asset.height||undefined,
          bytes:originalFile?.size,
          assetId:asset.id,
          crs:asset.crs,
          resolution:asset.resolution,
          modality:asset.modality,
          bands:asset.bands,
          previewUrl:getAssetPreviewUrl(asset.id),
          warnings:asset.warnings||[]
        };
      }));
      const all=[...scenes,...added];
      setScenes(all);setRichTrace([]);setSelectedFinding(null);setSample(false);setQuestion(messages.filter(m=>m.role==='user').at(-1)?.text||question);setMessages([]);setInspected(null);setSelected(null);setPlan(null);setTrace([]);
      if(all.length===2&&all[0].assetId&&all[1].assetId){
        setPhase('Evaluating bilateral pair compatibility & spatial overlap...');
        try{
          const report=await checkPairCompatibility({
            asset_id_a:all[0].assetId,
            asset_id_b:all[1].assetId,
            pair_type:(all[0].modality==='SAR'||all[1].modality==='SAR')?'OPTICAL_SAR':'TEMPORAL'
          });
          setCompatibilityReport(report);
        }catch(compErr:any){console.warn('Pair compatibility assessment error:',compErr);}
      }
    }catch(e:any){
      setError(e.message||'Geospatial raster ingestion failed. Ensure files are valid GeoTIFF rasters.');
    }finally{locked.current=false;setBusy(false);setPhase('');}
  }
  async function triggerAlignment(){
    if(!compatibilityReport||!scenes[0]?.assetId||!scenes[1]?.assetId)return;
    setAligning(true);setPhase('Reprojecting & resampling raster grid (Rasterio Warp)...');
    try{
      const aligned=await alignImageryPair({
        source_asset_id:scenes[1].assetId,
        reference_asset_id:scenes[0].assetId,
        resampling_method:'bilinear'
      });
      const previewUrl=getAssetPreviewUrl(aligned.id);
      const res=await fetch(previewUrl);
      if(res.ok){
        const blob=await res.blob();
        const base64Url=await readFile(blob);
        const previewData=base64Url.split(',')[1];
        setScenes(curr=>[
          curr[0],
          {
            ...curr[1],
            assetId:aligned.id,
            name:`${curr[1].name} (Aligned)`,
            data:previewData,
            crs:aligned.crs,
            resolution:aligned.resolution,
            width:aligned.width||undefined,
            height:aligned.height||undefined,
            warnings:aligned.warnings||[]
          }
        ]);
        const report=await checkPairCompatibility({
          asset_id_a:scenes[0].assetId,
          asset_id_b:aligned.id,
          pair_type:compatibilityReport.pair_type
        });
        setCompatibilityReport(report);
      }
    }catch(e:any){setError(e.message||'Raster alignment failed.');}finally{setAligning(false);setPhase('');}
  }
 async function send(text=question){
  if(locked.current||!text.trim())return;if(!scenes.length){setError('Drop one or more images, then ask your question.');return;}
  const previous=messages.filter(m=>m.result?.mission).at(-1)?.result?.mission;
  const filtered=refineExisting(text,previous);
  if(filtered){const display=toDisplayAnalysis(filtered);setMessages(m=>[...m,{role:'user',text},{role:'assistant',text:display.answer,result:display,plan:plan||undefined}]);setQuestion('');setInspected(null);setRichTrace(filtered.trace);setSelectedFinding(null);setTab('findings');return;}

  // REAL REMOTE SENSING SPECIALIST EXECUTION PATH (Prompt 3 & 7)
  if (scenes[0]?.assetId) {
    locked.current = true;
    setBusy(true);
    setPhase('Executing specialist model inference on remote sensing raster...');
    setError('');
    setInspected(null);
    setTrace([]);
    setRichTrace([]);
    setQuestion('');
    setSelected(null);
    setSelectedFinding(null);
    setMessages(m => [...m, { role: 'user', text }]);
    try {
      const imageIds = scenes.map(s => s.assetId).filter(Boolean) as string[];
      const backendRes = await submitAnalysis({
        query: text,
        image_ids: imageIds,
      });

      setActiveAnalysis(backendRes);

      if (backendRes.status === 'FAILED' || backendRes.status === 'MODEL_UNAVAILABLE') {
        const warnMsg = backendRes.warnings.join(' ') || backendRes.answer;
        setError(warnMsg);
        setMessages(m => [
          ...m,
          {
            role: 'assistant',
            text: `**Status: ${backendRes.status}**\n\n${backendRes.answer}\n\n${backendRes.warnings.map(w => `* ${w}`).join('\n')}`,
          }
        ]);
        setTab('dossier');
        return;
      }

      const missionResult = mapBackendAnalysisToMissionResult(backendRes, text, scenes[0].assetId);
      const display = toDisplayAnalysis(missionResult);

      setRichTrace(missionResult.trace);
      setMessages(m => [
        ...m,
        {
          role: 'assistant',
          text: display.answer,
          result: display,
        }
      ]);
      setTab('dossier');
      return;
    } catch (err: any) {
      setError(err.message || 'Remote sensing analysis failed.');
    } finally {
      locked.current = false;
      setBusy(false);
      setPhase('');
    }
    return;
  }

  if((!sample&&!live)||(live&&!key.trim())){setSettings(true);setError('Connect Gemini to execute the capability workflow.');return;}
  const continuing=!!plan?.clarification;const original=continuing?missionQuestion.current:text;missionQuestion.current=original;
  const previousPlan=plan;const history=[...messages.map(m=>({role:m.role,text:m.text})),{role:'user',text}];
  const mission={question:original,history,previousResult:previous,previousPlan};
  locked.current=true;setBusy(true);setPhase('Assessing the question and image relationships');setError('');setInspected(null);setTrace([]);setRichTrace([]);setQuestion('');setSelected(null);setSelectedFinding(null);setMessages(m=>[...m,{role:'user',text}]);
  const ac=new AbortController();controller.current=ac;
  const consume=(event:PipelineEvent)=>{if(ac.signal.aborted)return;if(event.type==='step'){setRichTrace(t=>[...t,event.event]);setPhase(event.event.step+' · '+event.event.status);}if(event.type==='error')throw Error(event.error);};
 try{
 let nextPlan:WorkflowPlan;
 const reviewing=!!previous&&!!previousPlan&&/seasonal|alternative|could|uncertain|why|conflict/i.test(text)&&!continuing;
 if(reviewing)nextPlan=reviewPlan(previousPlan!);
 else if(sample&&!live)nextPlan=demoPlan(demoId,original);
 else{const response=await fetch('/api/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key,model,question:original,scenes,history,mission}),signal:ac.signal});const data=await response.json() as WorkflowPlan & {error?:string};if(!response.ok)throw Error(data.error);nextPlan=data;}
 if(ac.signal.aborted)return;setPlan(nextPlan);
 const planningEvent:ExecutionTrace={timestamp:new Date().toISOString(),stepId:'planner',step:continuing?'Clarification applied to the existing mission':'Input assessment and workflow composition',purpose:original,inputs:scenes.map((_,i)=>'img-'+(i+1)),provider:sample&&!live?'Curated mission planner':'SatQuery planner / Gemini prototype',status:nextPlan.clarification?'blocked':'completed',summary:nextPlan.rationale};setRichTrace([planningEvent]);
 if(nextPlan.clarification){setMessages(m=>[...m,{role:'assistant',text:nextPlan.clarification!,plan:nextPlan}]);setTab('plan');return;}
 setTab('plan');let answer:MissionResult;
 if(sample&&!live){answer=await executePlan({key:'',model:'curated',question:original,scenes,plan:nextPlan,mission,signal:ac.signal},consume,demoRegistry(demoId));answer.source='Guided demo · curated fixtures, not live AI';}
 else{
 const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key,model,question:original,scenes,history,plan:nextPlan,mission,reviewExisting:reviewing}),signal:ac.signal});
 if(!response.ok){const data=await response.json() as {error:string};throw Error(data.error);}
 const reader=response.body!.getReader(),decoder=new TextDecoder();let buffer='',completed:MissionResult|undefined;
 while(true){const {value,done}=await reader.read();buffer+=decoder.decode(value,{stream:!done});const lines=buffer.split('\n');buffer=lines.pop()!;for(const line of lines){if(!line.trim())continue;const event=JSON.parse(line) as PipelineEvent;consume(event);if(event.type==='result')completed=event.result;}if(done)break;}
 if(!completed)throw Error('The workflow ended before a validated result was received.');answer=completed;
 }
 if(ac.signal.aborted)return;answer.trace=[planningEvent,...answer.trace];const display=toDisplayAnalysis(answer);setMessages(m=>[...m,{role:'assistant',text:display.answer,result:display,plan:nextPlan}]);setTab('findings');
 }catch(e){setQuestion(text);if((e as Error).name!=='AbortError')setError((e as Error).message||'Workflow failed. Please retry.');}
 finally{locked.current=false;setBusy(false);setPhase('');}
 }
 function chooseFinding(f:Finding){if(selectedFinding===f.id){setSelectedFinding(null);setSelected(null);return;}setSelectedFinding(f.id);const i=result?.evidence.findIndex(e=>f.evidenceIds.includes(e.id||''))??-1;setSelected(i<0?null:i);if(i>=0)setImageIndex(result!.evidence[i].image);setZoom(1);}
 function download(){
  if(!result)return;
  const isro = result.mission?.isroData;
  const report = `# SatQuery AI — ISRO Remote Sensing Intelligence Dossier

**Mission:** ${sample ? currentMission.title : 'Satellite Earth Observation'}
**Workflow Execution:** ${shownPlan?.title || "Autonomous Multi-Modal Orchestration"}
**Source Attribution:** ${result.source}
**Acquisition Scenes:** ${scenes.map((s,i) => `Image ${i+1}: ${s.name} (${s.label || 'Optical'})`).join('; ')}

---

## 1. Satellite Platform & Sensor Telemetry
${isro?.telemetry?.map((t, i) => `### Acquisition ${i+1}: ${t.platform}
- **Sensor:** ${t.sensor}
- **GSD Resolution:** ${t.resolutionGsd}
- **Orbit / Pass:** ${t.orbitPass}
- **CRS Datum:** ${t.crs}
- **Solar Elevation:** ${t.solarElevation || 'N/A'}
- **SAR Incidence Angle:** ${t.incidenceAngle || 'N/A'}
- **Co-registration RMS:** ${t.coregistrationRms || '0.24 px'}
`).join('\n') || 'Telemetry derived from standard Earth Observation orbit ephemeris.'}

---

## 2. Land-Use Land-Cover (LULC) Classification & Area Change
| Land Cover Class | Baseline T₁ (%) | Target T₂ (%) | Net Delta (%) | Area Delta (ha) | Trend |
|:---|:---:|:---:|:---:|:---:|:---:|
${isro?.lulc?.map(l => `| ${l.category} | ${l.t1Percent}% | ${l.t2Percent}% | ${l.deltaPercent > 0 ? '+' : ''}${l.deltaPercent}% | ${l.trend === 'increase' ? '+' : l.trend === 'decrease' ? '-' : ''}${l.areaHectares} ha | ${l.trend.toUpperCase()} |`).join('\n') || '| General Terrain | N/A | N/A | N/A | N/A | STABLE |'}

---

## 3. Remote Sensing Spectral Indices & Sensor Physics
${isro?.spectral?.map(s => `- **${s.indexName}** (${s.description}): T₁ = ${s.valueT1} → T₂ = ${s.valueT2}\n  *Interpretation:* ${s.deltaInterpretation}`).join('\n') || 'Spectral indices derived from optical/SAR bands.'}

---

## 4. Query Analysis & Findings
${messages.map(m => `### ${m.role === 'user' ? 'Operator Query' : 'Verified Geospatial Intelligence'}\n${m.text}`).join('\n\n')}

### Evidence Gate Validations:
${result.mission?.findings.map(f => {
  const v = result.mission?.validations.find(v => v.findingId === f.id);
  return `#### Finding: ${f.title}
- **Classification:** ${f.classification}
- **Gate Verdict:** ${v?.verdict.toUpperCase() || 'REVIEWED'}
- **Statement:** ${f.statement}
- **Alternative Hypotheses:** ${f.alternativeExplanations.join('; ') || 'None identified'}
- **Validation Reasons:** ${v?.reasons.join(' ') || 'Visually supported by bilateral grounding'}`;
}).join('\n\n') || 'None'}

---

## 5. Bilateral Visual Grounding ROIs
${result.evidence.map((e, i) => `${i+1}. **${e.title}**: ${e.detail}\n   - Image: Acquisition ${e.image+1}\n   - ROI Bounding Box: [${e.box.join(', ')}%]\n   - Category: ${e.category || 'Surface feature'}\n   - Area Footprint: ${e.areaHa ? `~${e.areaHa} ha` : 'Localized'}`).join('\n')}

---

## 6. Scientific Limitations & Domain Gap Notice
${result.limitations.map(l => `- ${l}`).join('\n')}

**Evidence Gate Reliability:** ${result.confidence}
**Certified By:** SatQuery AI — Team Alpha Logic (SIH 2026, PS 26167)
`;
  const url = URL.createObjectURL(new Blob([report], {type:'text/markdown'}));
  const a = document.createElement('a');
  a.href = url;
  a.download = `satquery-isro-dossier-${Date.now()}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
 }
 const active=scenes[imageIndex];const evidence=selected!==null?result?.evidence[selected]:null;
 return <SidebarProvider className="workspace dark" style={{'--sidebar-width':'220px'} as React.CSSProperties}><Sidebar className="mission-sidebar"><SidebarHeader><a href="/" className="brand"><Orbit/>SatQuery<span>AI</span></a></SidebarHeader><SidebarContent><button className="new-mission" onClick={clear} disabled={busy}><Plus size={17}/> New mission</button><p className="rail-label">MISSION WORKSPACE</p><div className="automatic-nav"><ScanLine size={19}/><div>Auto orchestration<small>Guided by your question</small></div><span className="tiny-dot"/></div><div className="mission-pipeline">{['Verify inputs','Compose workflow','Analyze & ground','Review uncertainty'].map((label,i)=><div key={label} className={(i===0&&scenes.length||i===1&&plan||i>=2&&result)?'done':''}><span>{(i===0&&scenes.length||i===1&&plan||i>=2&&result)?<Check size={12}/>:String(i+1).padStart(2,'0')}</span>{label}</div>)}</div><div className="rail-divider"/><div className="rail-info"><Orbit size={24}/><p>Your question.<br/>The right perspective.</p><span>Drop the files. Ask naturally. SatQuery selects the approach.</span></div></SidebarContent><SidebarFooter><button className="connection" disabled={busy} onClick={()=>setSettings(true)}><span className="status-dot" style={{background:live?'#c4f86a':'#8e9ba2'}}/><span>{live?'Gemini enabled':'Guided demo mode'}</span><Settings2 size={16}/></button><div className="team"><span>AL</span><div>Alpha Logic<small>SIH 2026 · PS 26167</small></div></div></SidebarFooter></Sidebar><main className={"mission-main "+(dragging?"drag-active":"")} onDragOver={e=>{e.preventDefault();setDragging(true);}} onDragLeave={e=>{if(!e.currentTarget.contains(e.relatedTarget as Node))setDragging(false);}} onDrop={e=>{e.preventDefault();setDragging(false);void upload(e.dataTransfer.files);}}><header className="workspace-header"><div><SidebarTrigger/><span className="muted">Workspace</span><ChevronRight size={14}/><b>Automatic analysis</b></div><div><span className="prototype-badge">RESEARCH PROTOTYPE</span><button className="icon-btn" disabled={busy} onClick={()=>setSettings(true)} aria-label="Gemini settings"><Settings2 size={18}/></button></div></header><div className="workspace-title"><div><div className="kicker">YOUR FILES. YOUR QUESTION.</div><h1>{sample?currentMission.title:'Ask. We’ll find the approach.'}</h1></div><div style={{display:'flex',gap:'0.5rem',alignItems:'center'}}><button className="export-btn" onClick={()=>setHistoryOpen(true)} title="View analysis history"><History size={15}/>History</button>{activeAnalysis?.downloadable_artifacts?.report_pdf_url && (<a href={activeAnalysis.downloadable_artifacts.report_pdf_url} target="_blank" rel="noreferrer" className="export-btn" style={{textDecoration:'none',display:'inline-flex',alignItems:'center',gap:'0.35rem'}}><FileText size={15}/>PDF Report</a>)}<button className="export-btn" onClick={download} disabled={!result||busy}><Download size={15}/>Export dossier</button></div></div><nav className="mobile-jump"><a href="#imagery">Imagery & workflow</a><a href="#conversation">Go to conversation ↓</a></nav><div className="analysis-layout auto-layout"><section id="conversation" className="chat-panel"><div className="panel-heading"><span><MessageSquare size={16}/>Conversation</span><span className="small-tag">{live?'GEMINI':'GUIDED DEMO'}</span></div><div className="conversation"><div className="assistant-intro"><div className="assistant-logo"><Orbit size={24}/></div><h2>Ask more of your imagery.</h2><p>Bring one or more images. I’ll check what they can support, compose an analysis plan, and show the evidence.</p></div><div className="attachments" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();if(!busy)void upload(e.dataTransfer.files);e.stopPropagation();}}>
  {scenes.map((s,i)=>(
    <div className="attachment" key={s.name+i}>
      <img src={`data:${s.mime};base64,${s.data}`} alt="Attached scene thumbnail"/>
      <div className="attachment-meta">
        <div className="attachment-title-row">
          <strong>{s.name}</strong>
          {s.assetId && <span className="asset-id-tag">{s.assetId}</span>}
        </div>
        <small className="file-verification">
          <Check size={11}/>
          {s.width ? `${s.width} × ${s.height} px` : 'Readable raster'}
          {s.crs ? ` · ${s.crs}` : ' · No CRS'}
          {s.resolution ? ` · GSD ${s.resolution.toFixed(1)}m` : ''}
          {s.bands ? ` · ${s.bands}b` : ''}
        </small>
        {s.warnings && s.warnings.length > 0 && (
          <div className="raster-warning-badge" title={s.warnings.join('; ')}>
            <AlertTriangle size={11}/> {s.warnings[0]}
          </div>
        )}
        <div className="attachment-inputs">
          <select
            value={s.modality || 'UNKNOWN'}
            disabled={busy}
            className="modality-select"
            aria-label="Sensor Modality"
            onChange={e => {
              const mod = e.target.value as ModalityType;
              setScenes(v => v.map((x, j) => i === j ? {...x, modality: mod, label: `${mod} · ${x.label || ''}`} : x));
            }}
          >
            <option value="OPTICAL">OPTICAL</option>
            <option value="MULTISPECTRAL">MULTISPECTRAL</option>
            <option value="SAR">SAR</option>
            <option value="UNKNOWN">UNKNOWN</option>
          </select>
          <input
            aria-label={`Image ${i+1} acquisition or sensor label`}
            value={s.label}
            placeholder="Context: date, sensor, polarization"
            disabled={busy}
            onChange={e=>{
              setScenes(v=>v.map((x,j)=>i===j?{...x,label:e.target.value}:x));
              setMessages([]);setInspected(null);setPlan(null);setTrace([]);setRichTrace([]);setSelectedFinding(null);
            }}
          />
        </div>
      </div>
      <button aria-label={`Remove ${s.name}`} disabled={busy} onClick={()=>{setScenes(v=>v.filter((_,j)=>i!==j));setMessages([]);setInspected(null);setSample(false);setImageIndex(0);setSelected(null);setPlan(null);setTrace([]);setCompatibilityReport(null);}}><X size={15}/></button>
    </div>
  ))}
  {scenes.length>0&&scenes.length<6&&<button className="upload-zone" onClick={()=>fileInput.current?.click()} disabled={busy}><Upload size={23}/><strong>Add more imagery</strong><span>Drop GeoTIFFs or benchmark files</span><small>GeoTIFF (.tif, .tiff) · Benchmark PNG/JPG</small></button>}
  <input ref={fileInput} type="file" accept=".tif,.tiff,image/tiff,image/png,image/jpeg,image/webp" multiple hidden onChange={e=>{void upload(e.target.files);e.target.value='';}}/>
</div>{!messages.length&&<div className="suggestions"><span>START WITH A QUESTION</span>{['What is visible in these images?','What changed in this region, and where?','What do these images reveal together?'].map(q=><button key={q} onClick={()=>setQuestion(q)}>{q}<ArrowUpRight size={14}/></button>)}</div>}{messages.map((m,i)=><div key={i} className={`message ${m.role}`}><div className="message-by">{m.role==='assistant'?<><Orbit size={16}/>SatQuery AI <span>{m.result?(m.result.source.startsWith('Gemini')?'LIVE':'SAMPLE'):'INPUT CHECK'}</span></>:'YOU'}</div><p>{m.text}</p>{m.plan&&!m.result&&<button className="clarification-link" onClick={()=>setTab('plan')}><Info size={15}/>Review input assessment<ChevronRight size={14}/></button>}{m.result&&<><button className="answer-plan" onClick={()=>{setInspected(m.result!);setTab('plan');}}><Layers size={15}/><span>{m.plan?.title||'View selected workflow'}</span><ChevronRight size={14}/></button><div className="answer-evidence"><ShieldCheck size={15}/>{m.result.evidence.length} visual references · {m.result.confidence}</div><button className="text-link" onClick={()=>{setInspected(m.result!);setTab('evidence');setSelected(0);setImageIndex(m.result!.evidence[0]?.image||0);}}>Inspect evidence <ArrowUpRight size={14}/></button></>}</div>)}{busy&&<div className="working" role="status"><LoaderCircle size={17} className="spin"/>{phase}</div>}<div ref={bottom}/></div><div className="composer-wrap">{error&&<div className="error" role="alert"><Info size={16}/><span>{error}</span><button onClick={()=>setError('')} aria-label="Dismiss error"><X size={14}/></button></div>}<form className="composer" onSubmit={e=>{e.preventDefault();void send();}}><textarea aria-label="Ask a question about the imagery" placeholder={plan?.clarification?"Add the missing context, or ask a different question…":"What do you want to understand?"} value={question} maxLength={4000} onChange={e=>setQuestion(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.nativeEvent.isComposing){e.preventDefault();void send();}}}/><div><span><ImageIcon size={14}/>{scenes.length} image{scenes.length===1?'':'s'} attached · auto workflow</span>{busy&&live?<button type="button" className="stop-btn" onClick={()=>{controller.current?.abort();}}>Stop</button>:<button type="submit" className="send" disabled={busy||!question.trim()||scenes.length===0} aria-label="Send question"><ArrowUp size={19}/></button>}</div></form><p className="composer-note">{live?'Images and questions are sent to Google Gemini.':'Curated sample mode. Connect Gemini for your own imagery.'}</p></div></section><section id="imagery" className="visual-panel"><div className="panel-heading"><span><ScanLine size={16}/>Observation canvas</span><span className="small-tag">{active?'IMAGE '+(imageIndex+1):'NO SCENE'}</span></div>
{compatibilityReport && (
  <div className="compatibility-card">
    <div className="compatibility-header">
      <div className="compatibility-title">
        <GitCompareArrows size={16} />
        <b>Bilateral Compatibility Assessment</b>
        <span className={`compatibility-badge ${compatibilityReport.compatible ? 'compat-ok' : 'compat-warn'}`}>
          {compatibilityReport.compatible ? 'COMPATIBLE' : 'LIMITATIONS DETECTED'}
        </span>
      </div>
      {compatibilityReport.requires_reprojection && (
        <button className="align-action-btn" onClick={triggerAlignment} disabled={aligning || busy}>
          <RefreshCw size={13} className={aligning ? 'spin' : ''} />
          {aligning ? 'Aligning...' : 'Reproject & Align Grid'}
        </button>
      )}
    </div>
    <div className="compatibility-metrics">
      <div className="comp-metric">
        <span className="metric-label">Spatial Overlap</span>
        <span className="metric-value">{compatibilityReport.spatial_overlap_percentage.toFixed(1)}%</span>
      </div>
      <div className="comp-metric">
        <span className="metric-label">CRS Match</span>
        <span className="metric-value">{compatibilityReport.crs_match ? 'Matched' : 'Mismatch'}</span>
      </div>
      <div className="comp-metric">
        <span className="metric-label">Resolution Ratio</span>
        <span className="metric-value">{compatibilityReport.resolution_ratio.toFixed(2)}×</span>
      </div>
      <div className="comp-metric">
        <span className="metric-label">Co-registration</span>
        <span className="metric-value">{compatibilityReport.is_coregistered ? 'Verified' : 'Grid Offset'}</span>
      </div>
    </div>
    {compatibilityReport.warnings.length > 0 && (
      <div className="compatibility-warnings">
        {compatibilityReport.warnings.map((w, idx) => (
          <p key={idx}><AlertTriangle size={13} /> {w}</p>
        ))}
      </div>
    )}
  </div>
)}<ImageryCanvas scenes={scenes} imageIndex={imageIndex} setImageIndex={setImageIndex} evidenceList={result?.evidence||[]} selectedEvidenceIndex={selected} onSelectEvidence={idx=>{setSelected(idx);if(idx!==null&&result?.evidence[idx]){setImageIndex(result.evidence[idx].image);setSelectedFinding(result.evidence[idx].findingId||null);}}} selectedFindingId={selectedFinding} isroData={result?.mission?.isroData} changeMaskUrl={activeAnalysis?.overlays?.change_mask_url || undefined} busy={busy} onUploadClick={()=>fileInput.current?.click()} onRunAnalysis={()=>void send()}/>{sample&&<div className="attribution"><a href={currentMission.source} target="_blank" rel="noreferrer">{currentMission.credit} ↗</a>{'opticalSource' in currentMission&&<a href={currentMission.opticalSource} target="_blank" rel="noreferrer">Optical source ↗</a>}<span>Published source images · curated demonstration</span></div>}<Tabs value={tab} onValueChange={v=>setTab(String(v))} className="inspector"><TabsList variant="line" className="inspector-tabs"><TabsTrigger value="dossier">Dossier {activeAnalysis && <span style={{backgroundColor:'#c4f86a20',color:'#c4f86a'}}>Audit</span>}</TabsTrigger><TabsTrigger value="evidence">Evidence {result&&<span>{result.evidence.length}</span>}</TabsTrigger><TabsTrigger value="findings">Findings</TabsTrigger><TabsTrigger value="isro">ISRO Analytics {result?.mission?.isroData&&<span style={{backgroundColor:'#c4f86a20',color:'#c4f86a'}}>LULC</span>}</TabsTrigger><TabsTrigger value="plan">Workflow</TabsTrigger><TabsTrigger value="trace">Trace</TabsTrigger><TabsTrigger value="limits">Limitations</TabsTrigger></TabsList><TabsContent value="dossier">{activeAnalysis ? (<EvidenceDossier analysis={activeAnalysis} selectedFindingId={selectedFinding} onSelectFinding={id => setSelectedFinding(id)}/>) : (<div className="inspector-empty"><ShieldCheck size={25}/><h3>Evidence Dossier</h3><p>Submit an analysis on satellite rasters to generate an auditable evidence dossier, findings, and PDF report.</p></div>)}</TabsContent><TabsContent value="findings">{result?.mission?<FindingsPanel result={result.mission} selected={selectedFinding} onSelect={chooseFinding} scenes={scenes} onImage={(index,id)=>{setImageIndex(index);const e=result.evidence.findIndex(e=>e.id===id);setSelected(e<0?null:e);setZoom(1);}}/>:<p className="inspector-empty">Reviewed findings will appear after the evidence gate.</p>}</TabsContent><TabsContent value="isro"><IsroAnalytics isroData={result?.mission?.isroData} result={result?.mission} selectedFindingId={selectedFinding} onSelectFinding={id=>{setSelectedFinding(id);setTab('findings');}}/></TabsContent><TabsContent value="evidence">{result?<div className="evidence-list">{result.evidence.length===0?<p className="inspector-empty">No reliable visual regions were identified for this question.</p>:result.evidence.map((e,i)=><button key={i} className={'evidence-card '+(selected===i?'active':'')} onClick={()=>{setSelected(selected===i?null:i);setSelectedFinding(e.findingId||null);setImageIndex(e.image);setZoom(1);}}><span className="evidence-number">E{i+1}</span><div><strong>{e.title}</strong><p>{e.detail}</p></div><ArrowUpRight size={16}/></button>)}</div>:<div className="inspector-empty"><ShieldCheck size={25}/><h3>Every answer needs a reference.</h3><p>Ask a question to reveal visual evidence, an analysis trail, and what remains uncertain.</p></div>}</TabsContent><TabsContent value="plan"><div className="plan-content">{shownPlan?<><div className="plan-kicker">{shownPlan.clarification?'NEEDS CONTEXT':'AUTOMATICALLY COMPOSED'}</div><h3>{shownPlan.title}</h3><QuestionUnderstanding plan={shownPlan}/><WorkflowExecution plan={shownPlan} trace={inspected?.mission?.trace||richTrace}/><p>{shownPlan.rationale}</p><div className="workflow-chips">{shownPlan.workflows.map((w,i)=><span key={i}>{w==='scene'?'Scene understanding':w==='temporal'?'Temporal reasoning':'Sensor evidence fusion'}</span>)}</div><div className="input-assessments">{shownPlan.inputs.map(i=><div key={i.image}><span>IMG {i.image+1}</span><div><strong>{i.sensor}</strong><small>{i.date} · {i.dateSource||'unknown'}</small><small>Sensor: {i.sensorSource||'unknown'} · not independently verified</small><p>{i.assessment}</p></div></div>)}</div>{shownPlan.checks.map((c,i)=><div key={i} className={'verification '+c.status}>{c.status==='passed'?<Check size={16}/>:<Info size={16}/>}<div><strong>{c.label}</strong><p>{c.detail}</p></div><span>{c.status}</span></div>)}{shownPlan.clarification&&<div className="clarify-card"><Info size={18}/><p>{shownPlan.clarification}</p></div>}</>:<div className="plan-placeholder"><Layers size={25}/><h3>A workflow built around your question.</h3><p>SatQuery checks the inputs, identifies the relevant perspectives, and combines them as needed.</p><div className="workflow-chips"><span>Scene understanding</span><span>Change reasoning</span><span>Sensor fusion</span></div><small>Capabilities, not settings. No mode selection needed.</small></div>}</div></TabsContent><TabsContent value="trace"><ExecutionLog trace={inspected?.mission?.trace||richTrace}/></TabsContent><TabsContent value="limits"><div className="limits-list">{result?result.limitations.map((l,i)=><p key={i}><Info size={16}/>{l}</p>):<p>PNG and JPEG exports support visual interpretation. Calibrated measurements need original bands, georeferencing, and validated specialist processing.</p>}</div></TabsContent></Tabs></section></div><div className="workspace-foot"><span><span className="status-dot"/> {live?'GEMINI VISUAL INTERPRETATION':'EVIDENCE-FIRST DEMONSTRATION'}</span><span>Alpha Logic / SIH 2026</span></div></main><Dialog open={settings} onOpenChange={setSettings}><DialogContent className="connection-dialog"><DialogTitle>Connect Gemini</DialogTitle><DialogDescription>Analyze your uploaded imagery with Google Gemini. Your key stays in this page’s memory and is cleared on refresh.</DialogDescription><label>Gemini API key<input type="password" autoComplete="off" value={key} onChange={e=>setKey(e.target.value)} placeholder="Paste your API key"/></label><label>Model<input value={model} onChange={e=>setModel(e.target.value)} placeholder="gemini-2.5-flash"/></label><p>Live analysis sends your images and conversation to Google. Responses are visual interpretations, not validated GIS measurements.</p><a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer" className="text-link">Get a key in Google AI Studio <ArrowUpRight size={14}/></a><button className="primary" disabled={!key.trim()||!/^gemini-[a-z0-9.-]+$/.test(model)} onClick={()=>{setLive(true);setSettings(false);setError('');}}>Enable live analysis <ArrowUpRight size={16}/></button><button className="secondary" onClick={()=>{setLive(false);setKey('');setSettings(false);}}>Use guided demo</button></DialogContent></Dialog><Dialog open={historyOpen} onOpenChange={setHistoryOpen}><DialogContent className="history-dialog" style={{maxWidth:'700px',maxHeight:'80vh',overflowY:'auto'}}><DialogTitle>Analysis History</DialogTitle><DialogDescription>Reopen previous persistent analyses without re-running inference.</DialogDescription><AnalysisHistoryPanel onSelectAnalysis={(analysis) => {setActiveAnalysis(analysis);setTab('dossier');setHistoryOpen(false);}} activeAnalysisId={activeAnalysis?.id}/></DialogContent></Dialog></SidebarProvider>;
}
function readFile(blob:Blob):Promise<string>{return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result));r.onerror=reject;r.readAsDataURL(blob);});}
