import {gemini,constraints} from './gemini';
import type {Scene} from './analysis';
import type {WorkflowPlan} from './workflow';
import type {WorkflowCapability,WorkflowStep,CandidateBundle,EvidenceValidation,MissionContext} from './domain';
import scene from './prompts/scene-analysis';
import temporal from './prompts/temporal-analysis';
import optical from './prompts/optical-analysis';
import sar from './prompts/sar-analysis';
import cross from './prompts/cross-sensor-analysis';
import grounding from './prompts/visual-grounding';
import uncertainty from './prompts/uncertainty-check';
import validation from './prompts/evidence-validation';
import synthesis from './prompts/synthesis';
export type ToolExecutionContext={scenes:Scene[];question:string;plan:WorkflowPlan;step:WorkflowStep;mission?:MissionContext;candidates:CandidateBundle;validations:EvidenceValidation[];key:string;model:string;signal?:AbortSignal};
export interface CapabilityProvider {name:string;execute(context:ToolExecutionContext):Promise<unknown>}
export interface SceneAnalysisProvider extends CapabilityProvider{}
export interface TemporalAnalysisProvider extends CapabilityProvider{}
export interface CrossSensorAnalysisProvider extends CapabilityProvider{}
export interface GroundingProvider extends CapabilityProvider{}
export interface EvidenceValidationProvider extends CapabilityProvider{}
export type CapabilityRegistry=Record<WorkflowCapability,CapabilityProvider>;
const schema=`Return JSON {findings:[{id:string,title:string,statement:string,category:string,classification:"observation|likely_interpretation|uncertain",evidenceIds:[string],confidence:{level:"high|medium|low",meaning:"AI-assessed visual support, not calibrated"},alternativeExplanations:[string],limitations:[string]}],evidenceItems:[{id:string,imageId:string,findingId:string,evidenceType:"visual_region|temporal_difference|cross_sensor_support",description:string,region?:{x:number,y:number,width:number,height:number},reliability:"high|medium|low",limitations:[string]}],crossSensorAssessment:[{finding:string,relationship:"agreement|complementary|disagreement|inconclusive",explanation:string,evidenceIds:[string]}]}. Regions use percentages 0–100 and must stay inside the image. Every evidence item belongs to exactly one finding and is referenced by it. Every reference must exist. Produce at most 6 findings and 12 evidence items. Do not fabricate evidence to fill the schema.`;
const prompts:Record<WorkflowCapability,string>={scene_understanding:scene,temporal_comparison:temporal,optical_analysis:optical,sar_analysis:sar,cross_sensor_analysis:cross,visual_grounding:grounding,uncertainty_check:uncertainty,evidence_validation:validation,synthesis};
export const registry=Object.fromEntries(Object.entries(prompts).map(([capability,instructions])=>[capability,{name:`Gemini prototype · ${capability.replaceAll('_',' ')}`,async execute(c:ToolExecutionContext){
 const indexed=c.scenes.map((s,i)=>({...s,imageId:`img-${i+1}`})).filter(s=>c.step.inputImageIds.includes(s.imageId));
 const shape=['evidence_validation','synthesis'].includes(capability)?'':schema;
 const prompt=`${constraints}\nCAPABILITY: ${capability}\n${instructions}\n${shape}\nQuestion: ${c.question}\nStep: ${JSON.stringify(c.step)}\nInput order and metadata: ${JSON.stringify(indexed.map(({data,...s})=>s))}\nRelationships and assessments: ${JSON.stringify({inputs:c.plan.inputs,relationships:c.plan.inferredRelationships})}\nPrior stage candidates: ${JSON.stringify(c.candidates)}\nGate verdicts: ${JSON.stringify(c.validations)}\nPrevious mission context: ${JSON.stringify(c.mission||{}).slice(0,120000)}`;
 return gemini(c.key,c.model,prompt,indexed,c.signal);
 }}])) as CapabilityRegistry;
