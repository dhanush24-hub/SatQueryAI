import {assertExecutable,type WorkflowPlan} from './workflow';
import {registry,type CapabilityRegistry,type ToolExecutionContext} from './registry';
import {validateBundle,validateVerdicts,gateFindings,validateSynthesis} from './validation';
import type {CandidateBundle,ExecutionTrace,MissionResult,EvidenceValidation} from './domain';
export type PipelineEvent={type:'step';event:ExecutionTrace}|{type:'result';result:MissionResult}|{type:'error';error:string};
function merge(a:CandidateBundle,b:CandidateBundle,prefix:string):CandidateBundle{
 const f=(id:string)=>prefix+'-'+id,e=(id:string)=>prefix+'-'+id;
 return {findings:[...a.findings,...b.findings.map(x=>({...x,id:f(x.id),evidenceIds:x.evidenceIds.map(e)}))],evidenceItems:[...a.evidenceItems,...b.evidenceItems.map(x=>({...x,id:e(x.id),findingId:f(x.findingId)}))],crossSensorAssessment:[...a.crossSensorAssessment,...b.crossSensorAssessment.map(x=>({...x,evidenceIds:x.evidenceIds.map(e)}))]};
}
export async function executePlan(context:Omit<ToolExecutionContext,'step'|'candidates'|'validations'>,emit:(event:PipelineEvent)=>void=()=>{},providers:CapabilityRegistry=registry):Promise<MissionResult>{
 const plan:WorkflowPlan=structuredClone(context.plan);assertExecutable(plan);
 let candidates:CandidateBundle=context.mission?.previousResult&&!plan.workflow.some(s=>['scene_understanding','temporal_comparison','optical_analysis','sar_analysis','cross_sensor_analysis'].includes(s.capability))?structuredClone({findings:context.mission.previousResult.findings,evidenceItems:context.mission.previousResult.evidenceItems,crossSensorAssessment:context.mission.previousResult.crossSensorAssessment}):{findings:[],evidenceItems:[],crossSensorAssessment:[]},validations:EvidenceValidation[]=[];
 let final={directAnswer:'No supported findings are available.',uncertainties:[] as string[],nextQuestion:''};
 const trace:ExecutionTrace[]=[];const allIds=context.scenes.map((_,i)=>`img-${i+1}`);
 for(const step of plan.workflow){
 context.signal?.throwIfAborted();if(step.dependencies.some(id=>plan.workflow.find(s=>s.id===id)?.status!=='completed'))throw Error('A workflow dependency did not complete.');
 const provider=providers[step.capability];if(!provider)throw Error('The requested capability has no registered provider.');
 const record=(status:ExecutionTrace['status'],summary:string)=>{const event={timestamp:new Date().toISOString(),stepId:step.id,step:step.title,purpose:step.reason,inputs:step.inputImageIds,provider:provider.name,status,summary};trace.push(event);emit({type:'step',event});};
 step.status='running';record('running','Provider request started.');
 try{
 const raw=await provider.execute({...context,plan,step,candidates,validations});
 if(step.capability==='evidence_validation'){validations=validateVerdicts(raw,candidates,plan);candidates=gateFindings(candidates,validations);record('completed',`${validations.filter(v=>v.verdict==='supported').length} supported; ${validations.filter(v=>v.verdict!=='supported').length} qualified, conflicting or insufficient. No independent model verification.`);}
 else if(step.capability==='synthesis'){if(validations.length!==candidates.findings.length)throw Error('The evidence gate is incomplete.');final=validateSynthesis(raw,candidates,validations);record('completed','Answer synthesized from gated findings.');}
 else {
 const bundle=validateBundle(raw,step.inputImageIds,step.id);
 if(['visual_grounding','uncertainty_check'].includes(step.capability)){
 const priorIds=new Set(candidates.findings.map(f=>f.id));if(bundle.findings.some(f=>!priorIds.has(f.id)))throw Error('Review stage introduced an untracked finding.');
 const rank={low:0,medium:1,high:2};bundle.findings=bundle.findings.map(f=>{const prior=candidates.findings.find(p=>p.id===f.id);return prior&&rank[f.confidence.level]>rank[prior.confidence.level]?{...f,confidence:prior.confidence,classification:prior.classification}:f;});
 bundle.evidenceItems=bundle.evidenceItems.map(e=>({...e,sourceStep:candidates.evidenceItems.find(p=>p.id===e.id)?.sourceStep||e.sourceStep}));const omitted=candidates.findings.filter(f=>!bundle.findings.some(b=>b.id===f.id));for(const f of omitted){bundle.findings.push({...f,classification:'uncertain',confidence:{...f.confidence,level:'low'},limitations:[...f.limitations,'Not retained by the review; evidence is unresolved.']});bundle.evidenceItems.push(...candidates.evidenceItems.filter(e=>f.evidenceIds.includes(e.id)));}candidates=bundle;
 }else candidates=merge(candidates,bundle,step.id);
 // Bound accumulated output before later reasoning stages.
 if(candidates.findings.length>36||candidates.evidenceItems.length>72)throw Error('The analysis produced too many candidates. Narrow the question.');
 record('completed',`${bundle.findings.length} candidate findings; ${bundle.evidenceItems.length} approximate visual references.`);
 }
 step.status='completed';
 }catch(e){step.status='blocked';record('failed',e instanceof Error?e.message:'Provider response failed.');throw e;}
 }
 const result:MissionResult={...candidates,...final,validations,trace,source:'Gemini capability adapters · self-review, not independent validation',workflow:plan.workflow};emit({type:'result',result});return result;
}
