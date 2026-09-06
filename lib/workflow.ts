import type {Scene} from './analysis';
export type WorkflowPlan={title:string;rationale:string;workflows:('scene'|'temporal'|'fusion')[];inputs:{image:number;sensor:string;date:string;assessment:string}[];checks:{label:string;status:'passed'|'uncertain'|'blocked';detail:string}[];steps:string[];clarification:string|null};
export function validatePlan(value:unknown,count:number):WorkflowPlan{
 const p=value as WorkflowPlan;
 if(!p||typeof p.title!=='string'||typeof p.rationale!=='string'||!Array.isArray(p.workflows)||!p.workflows.length||p.workflows.some(w=>!['scene','temporal','fusion'].includes(w))||!Array.isArray(p.inputs)||p.inputs.length!==count||new Set(p.inputs.map(i=>i.image)).size!==count||p.inputs.some(i=>!Number.isInteger(i.image)||i.image<0||i.image>=count||typeof i.sensor!=='string'||typeof i.date!=='string'||typeof i.assessment!=='string')||!Array.isArray(p.checks)||p.checks.some(c=>!['passed','uncertain','blocked'].includes(c.status)||typeof c.label!=='string'||typeof c.detail!=='string')||!Array.isArray(p.steps)||!p.steps.length||p.steps.some(s=>typeof s!=='string')||(p.clarification!==null&&typeof p.clarification!=='string'))throw Error('The input assessment was incomplete. Please try again.');
 if(count<2&&p.workflows.some(w=>w==='temporal'||w==='fusion'))p.clarification='Please add another acquisition or complementary sensor image so I can answer this question.';
 if(p.checks.some(c=>c.status==='blocked')&&!p.clarification)p.clarification='Please resolve the flagged input issue before analysis can continue.';
 return {...p,title:p.title.slice(0,120),rationale:p.rationale.slice(0,1500),steps:p.steps.slice(0,6),checks:p.checks.slice(0,8)};
}
export function samplePlan(question:string):WorkflowPlan{
 const temporal=/chang|before|after|loss|increase|decrease/i.test(question);
 const quantitative=/ndvi|hectare|area|percent|count/i.test(question);
 return {title:temporal?'Temporal comparison needs another acquisition':quantitative?'Measurement needs original data':'Scene understanding + visual grounding',rationale:temporal?'Your question asks about change. This file is a mosaic, not a dated before-and-after pair.':quantitative?'A display image does not contain the spectral or geospatial information needed for a defensible measurement.':'One archival optical mosaic supports qualitative observations of visible features.',workflows:temporal?['temporal']:['scene'],inputs:[{image:0,sensor:'Optical · source documented',date:'1999–2000 mosaic',assessment:'Archival Landsat 7 display image'}],checks:[{label:'Input readability',status:'passed',detail:'A supported display image is loaded.'},{label:'Acquisition suitability',status:temporal||quantitative?'blocked':'uncertain',detail:temporal?'No independent second acquisition is present.':quantitative?'Original bands and georeferencing are absent.':'The mosaic is suitable for visual description, not current conditions.'},{label:'Geospatial verification',status:'uncertain',detail:'CRS, pixel scale and co-registration have not been verified.'}],steps:['Inspect visible land and water patterns','Locate approximate visual evidence','Report uncertainty and unsupported claims'],clarification:temporal?'Please add a separate, dated image of this region, or ask what is visible in this scene.':quantitative?'Please provide the required original spectral/geospatial data to a specialist pipeline, or ask for a qualitative visual assessment.':null};
}
export function validateScenes(scenes:unknown):asserts scenes is Scene[]{
 if(!Array.isArray(scenes)||scenes.length<1||scenes.length>6||scenes.some(s=>!s||typeof s.name!=='string'||!['image/jpeg','image/png','image/webp'].includes(s.mime)||typeof s.data!=='string'||s.data.length>7000000||!/^[A-Za-z0-9+/=]+$/.test(s.data))||scenes.reduce((n,s)=>n+s.data.length,0)>14000000)throw Error('Attach 1–6 PNG, JPG or WebP images: 5 MB each, 10 MB total.');
 if(scenes.some(s=>!matchesImageSignature(s.mime,s.data)))throw Error('An image signature does not match its format. Use a valid PNG, JPG or WebP file.');
 if(new Set(scenes.map(s=>s.data)).size!==scenes.length)throw Error('Duplicate images detected. Remove the duplicate acquisition before continuing.');
}

export function matchesImageSignature(mime:string,data:string){
 try{const bytes=atob(data.slice(0,32));return mime==='image/png'?bytes.startsWith('\x89PNG\r\n\x1a\n'):mime==='image/jpeg'?bytes.startsWith('\xff\xd8\xff'):mime==='image/webp'?bytes.startsWith('RIFF')&&bytes.slice(8,12)==='WEBP':false;}catch{return false;}
}
