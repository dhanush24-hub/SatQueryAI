import type {Scene} from './analysis';
import type {WorkflowStep,InputAssessment,ImageRelationship} from './domain';
export type WorkflowPlan={title:string;rationale:string;intent:string;intents:string[];requiredContext:string[];clarificationRequired:boolean;clarificationQuestions:string[];inferredRelationships:ImageRelationship[];workflow:WorkflowStep[];workflows:('scene'|'temporal'|'fusion')[];inputs:InputAssessment[];checks:{label:string;status:'passed'|'uncertain'|'blocked';detail:string}[];steps:string[];clarification:string|null;relevantImageIds:string[]};
const intentPatterns:Record<string,RegExp>={scene_description:/describ|visible|what.*see/i,object_detection:/building|object|structure/i,land_cover_reasoning:/land.cover|agricultur|vegetation/i,infrastructure_detection:/settlement|road|construct|built|infrastruct/i,temporal_change:/chang|expand|increase|decrease|before|after/i,vegetation_change:/vegetation.*(loss|chang)|deforest/i,water_change:/water.*(extent|chang)/i,flood_reasoning:/flood|inundat/i,cross_sensor_confirmation:/SAR|radar|sensor|confirm.*observation/i,cross_sensor_disagreement:/disagree|contradict/i,spatial_localization:/where|locat/i};
export function classifyIntent(question:string){const intents=Object.entries(intentPatterns).filter(([,p])=>p.test(question)).map(([i])=>i);return intents.length?intents:['uncertain_other'];}
const unique=<T,>(v:T[])=>[...new Set(v)];
const arrayStrings=(v:unknown)=>Array.isArray(v)&&v.every(x=>typeof x==='string')?v as string[]:[];
export function validatePlan(value:unknown,count:number,context=''):WorkflowPlan{
 const p=value as Record<string,any>;
 if(!p||typeof p.title!=='string'||typeof p.rationale!=='string'||!Array.isArray(p.inputs)||p.inputs.length!==count)throw Error('The input assessment was incomplete. Please try again.');
 const ids=Array.from({length:count},(_,i)=>`img-${i+1}`);
 const inputs:InputAssessment[]=p.inputs.map((i:any,index:number)=>{
 if(!i||i.image!==index||typeof i.sensor!=='string'||typeof i.date!=='string'||typeof i.assessment!=='string')throw Error('Input assessment must cover every image in order.');
 const provenance=(v:unknown)=>['provided','inferred','unknown'].includes(String(v))?v:'unknown';
 let dateSource=provenance(i.dateSource),sensorSource=provenance(i.sensorSource);
 if(dateSource==='provided'&&!context.includes(i.date))dateSource='inferred';
 if(sensorSource==='provided'&&!/sar|radar|optical|sentinel|landsat/i.test(context))sensorSource='inferred';
 if(i.sensorKind==='sar'&&sensorSource==='provided'){
 let label='',userText=context,briefProof=false;
 try{const c=JSON.parse(context);label=c.labels?.[index]||'';const history=Array.isArray(c.history)?c.history:[];userText=history.filter((m:any)=>m.role==='user').map((m:any)=>m.text).join(' ');const latest=history.at(-1)?.text||'';const pending=history.slice(0,-1).reverse().find((m:any)=>m.role==='assistant')?.text||'';briefProof=/which.*(?:SAR|radar)/i.test(pending)&&new RegExp('^(?:the )?(?:image |img[- ]?)?'+(index===0?'(?:1|first)':index===1?'(?:2|second)':String(index+1))+'(?: image| one| observation)?[.!]?$','i').test(latest.trim());}catch{}
 const name='(?:image|img)[ -]?'+(index+1);const linked=new RegExp(name+'[^.!?;]{0,50}(?:SAR|radar|Sentinel[- ]?1)|(?:SAR|radar)[^.!?;]{0,30}'+name,'i').test(userText);
 if(!/sar|radar|sentinel[- ]?1/i.test(label)&&!linked&&!briefProof)sensorSource='inferred';
 }
 return {...i,imageId:ids[index],sensorKind:['optical','sar','unknown'].includes(i.sensorKind)?i.sensorKind:'unknown',dateSource,sensorSource};
 });
 const intents=unique([...arrayStrings(p.intents),...classifyIntent(p.intent||context)]).slice(0,15);
 const relationships:ImageRelationship[]=(Array.isArray(p.inferredRelationships)?p.inferredRelationships:[]).map((r:any)=>{
 if(!r||!ids.includes(r.imageA)||!ids.includes(r.imageB)||r.imageA===r.imageB||!['temporal','cross_sensor','same_scene','possibly_unrelated','unknown'].includes(r.relationship)||!['high','medium','low'].includes(r.confidence))throw Error('Malformed image relationship.');
 return {imageA:r.imageA,imageB:r.imageB,relationship:r.relationship,confidence:r.confidence,basis:arrayStrings(r.basis),verified:false,earlierImageId:[r.imageA,r.imageB].includes(r.earlierImageId)?r.earlierImageId:undefined};
 });
 const relevant=unique(arrayStrings(p.relevantImageIds).filter(id=>ids.includes(id)));if(!relevant.length)relevant.push(...ids);
 const questions=arrayStrings(p.clarificationQuestions).slice(0,2);
 const temporal=intents.some(i=>['temporal_change','construction_change','vegetation_change','water_change'].includes(i));
 const fusion=intents.some(i=>['cross_sensor_confirmation','cross_sensor_disagreement'].includes(i));
 const selected=inputs.filter(i=>relevant.includes(i.imageId));
 const step=(capability:WorkflowStep['capability'],title:string,reason:string,inputImageIds:string[],dependencies:string[]=[]):WorkflowStep=>({id:`step-${workflow.length+1}`,capability,title,reason,inputImageIds,dependencies,status:'pending'});
 const workflow:WorkflowStep[]=[];
 const temporalIds=unique(relationships.filter(r=>r.relationship==='temporal').flatMap(r=>[r.imageA,r.imageB]));
 const temporalSelected=temporalIds.length?selected.filter(i=>temporalIds.includes(i.imageId)):fusion?selected.filter(i=>i.sensorKind==='optical'):selected;
 const dated=temporalSelected.every(i=>i.dateSource==='provided'&&/^\d{4}-\d{2}-\d{2}$/.test(i.date)&&!Number.isNaN(Date.parse(i.date)))&&new Set(temporalSelected.map(i=>i.date)).size===temporalSelected.length;
 let explicitOrder:RegExpMatchArray|undefined=[...context.matchAll(/(?:image|img)[ -]?(\d)\s+(?:is |was )?(?:the )?(?:earlier|first|before)/gi)].at(-1)||[...context.matchAll(/(?:the )?(first|second) (?:image|one|observation) (?:is |was )?(?:earlier|first)/gi)].at(-1);
 try{const c=JSON.parse(context);const latest=c.history?.at(-1)?.text;const pending=c.history?.slice(0,-1).reverse().find((m:any)=>m.role==='assistant')?.text;if(typeof latest==='string'&&/earlier|chronolog/i.test(pending||'')){const brief=latest.trim().match(/^(?:the )?(?:image |img[- ]?)?(1|2|first|second)(?: image| one| observation)?[.!]?$/i);if(brief)explicitOrder=brief as RegExpMatchArray;}}catch{}
 let ordered=dated?[...temporalSelected].sort((a,b)=>a.date.localeCompare(b.date)):temporalSelected;
 if(explicitOrder&&temporalSelected.length===2){const first=explicitOrder[1]==='first'?1:explicitOrder[1]==='second'?2:Number(explicitOrder[1]);if(dated&&ordered[0]?.image!==first-1)questions.push('The supplied dates conflict with the stated order. Which chronology should I use?');ordered=[...temporalSelected].sort((a,b)=>Number(b.image===first-1)-Number(a.image===first-1));}
 if(temporal){
 if(temporalSelected.length<2)questions.push('Please add another acquisition of this region to assess change.');
 else if(!dated&&!explicitOrder)questions.push(temporalSelected.length===2?'Which observation was captured earlier? For example, “Image 1 is earlier.”':'What is the chronological order of these observations? Add acquisition dates to the image context.');
 const relevantRelations=relationships.filter(r=>relevant.includes(r.imageA)&&relevant.includes(r.imageB));
 if(relevantRelations.some(r=>r.relationship==='possibly_unrelated'))questions.push('These observations may show different places. Which images show the same region?');
 if(!relevantRelations.some(r=>r.confidence!=='low'&&['temporal','same_scene'].includes(r.relationship))&&!/same (region|area|location|scene)/i.test(context))questions.push('Do these observations show the same region?');
 for(let i=0;i<ordered.length-1;i++)workflow.push(step('temporal_comparison',`Compare ${ordered[i].imageId} → ${ordered[i+1].imageId}`,'Identify change candidates while considering acquisition and seasonal differences.',[ordered[i].imageId,ordered[i+1].imageId]));
 }
 if(fusion){
 const optical=selected.filter(i=>i.sensorKind==='optical'),sar=selected.filter(i=>i.sensorKind==='sar'&&i.sensorSource==='provided');
 if(!sar.length)questions.push('Which image is the SAR observation? Sensor identity cannot be established from grayscale appearance alone.');
 if(!optical.length)questions.push('Which image is the optical observation?');
 if(optical.length)workflow.push(step('optical_analysis','Extract optical evidence','Preserve optical observations separately before fusion.',optical.map(i=>i.imageId)));
 if(sar.length)workflow.push(step('sar_analysis','Inspect radar display evidence','Interpret only the user-identified radar display, without calibrated backscatter claims.',sar.map(i=>i.imageId)));
 const sourceSteps=workflow.map(s=>s.id);workflow.push(step('cross_sensor_analysis','Compare sensor contributions','Evaluate agreement, complementarity, disagreement and missing support.',relevant,sourceSteps));
 }
 if(!temporal&&!fusion)workflow.push(step('scene_understanding','Understand the supplied scenes','Describe independently visible features; relationships are not required for description.',relevant));
 const analysisIds=workflow.map(s=>s.id);
 workflow.push(step('visual_grounding','Review visual references','Inspect approximate regions and link each to its finding.',relevant,analysisIds));
 workflow.push(step('uncertainty_check','Challenge candidate explanations','Test seasonal, acquisition and interpretation alternatives.',relevant,[workflow.at(-1)!.id]));
 workflow.push(step('evidence_validation','Apply the evidence gate','Give every candidate an explicit support verdict.',relevant,[workflow.at(-1)!.id]));
 workflow.push(step('synthesis','Synthesize the validated answer','Answer from reviewed findings, preserving conflicting and insufficient evidence.',relevant,[workflow.at(-1)!.id]));
 const checks=Array.isArray(p.checks)?p.checks.filter((c:any)=>c&&['passed','uncertain','blocked'].includes(c.status)&&typeof c.label==='string'&&typeof c.detail==='string'):[];
 for(const c of checks)if(c.status==='blocked'&&!questions.length)questions.push(c.detail);
 const clarificationQuestions=unique(questions).slice(0,2);const clarification=clarificationQuestions[0]||null;
 return {title:p.title.slice(0,160),rationale:p.rationale.slice(0,2000),intent:typeof p.intent==='string'?p.intent:context.slice(0,400),intents,requiredContext:arrayStrings(p.requiredContext),clarificationRequired:!!clarification,clarificationQuestions,inferredRelationships:relationships,workflow:workflow.map(s=>({...s,status:clarification?'blocked':'pending'})),workflows:[...(temporal?['temporal' as const]:[]),...(fusion?['fusion' as const]:[]),...(!temporal&&!fusion?['scene' as const]:[])],inputs,checks,steps:workflow.map(s=>s.title),clarification,relevantImageIds:relevant};
}
export function assertExecutable(plan:WorkflowPlan){
 if(plan.clarificationRequired||plan.clarification)throw Error('Resolve the clarification before executing the workflow.');
 const done=new Set<string>();for(const s of plan.workflow){if(done.has(s.id)||s.dependencies.some(d=>!done.has(d)))throw Error('Workflow dependencies are invalid or cyclic.');done.add(s.id);}
 if(plan.workflow.at(-1)?.capability!=='synthesis'||!plan.workflow.some(s=>s.capability==='evidence_validation'))throw Error('An evidence gate is required before synthesis.');
}
export function samplePlan(question:string):WorkflowPlan{return validatePlan({title:'Scene understanding + evidence gate',rationale:'A single archival mosaic supports visual interpretation only.',intent:question,intents:classifyIntent(question),inputs:[{image:0,sensor:'Landsat optical',sensorKind:'optical',sensorSource:'provided',date:'1999–2000 mosaic',dateSource:'provided',assessment:'Archival display mosaic; not a dated temporal pair.'}],checks:[],clarificationQuestions:[],inferredRelationships:[]},1,question+' Landsat optical 1999–2000 mosaic');}
export function validateScenes(scenes:unknown):asserts scenes is Scene[]{
 if(!Array.isArray(scenes)||scenes.length<1||scenes.length>6||scenes.some(s=>!s||typeof s.name!=='string'||!['image/jpeg','image/png','image/webp'].includes(s.mime)||typeof s.data!=='string'||s.data.length>7000000||!/^[A-Za-z0-9+/=]+$/.test(s.data))||scenes.reduce((n,s)=>n+s.data.length,0)>14000000)throw Error('Attach 1–6 PNG, JPG or WebP images: 5 MB each, 10 MB total.');
 if(scenes.some(s=>!matchesImageSignature(s.mime,s.data)))throw Error('An image signature does not match its format. Use a valid PNG, JPG or WebP file.');
 if(new Set(scenes.map(s=>s.data)).size!==scenes.length)throw Error('Duplicate images detected. Remove the duplicate acquisition before continuing.');
}

export function matchesImageSignature(mime:string,data:string){
 try{const bytes=atob(data.slice(0,32));return mime==='image/png'?bytes.startsWith('\x89PNG\r\n\x1a\n'):mime==='image/jpeg'?bytes.startsWith('\xff\xd8\xff'):mime==='image/webp'?bytes.startsWith('RIFF')&&bytes.slice(8,12)==='WEBP':false;}catch{return false;}
}
