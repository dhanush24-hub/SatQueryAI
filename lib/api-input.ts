import {validateScenes} from './workflow';
export async function readInput(request:Request){
 if(request.headers.get('origin')&&new URL(request.headers.get('origin')!).origin!==new URL(request.url).origin)throw Error('Request origin not allowed.');
 const raw=await request.text();if(raw.length>15500000)throw Error('Files and context exceed the request size limit.');
 let v;try{v=JSON.parse(raw);}catch{throw Error('The request could not be read.');}
 validateScenes(v.scenes);
 if(typeof v.key!=='string'||!v.key.trim()||typeof v.model!=='string'||!/^gemini-[a-z0-9.-]+$/.test(v.model)||typeof v.question!=='string'||!v.question.trim()||v.question.length>4000)throw Error('Connect Gemini and enter a question of up to 4,000 characters.');
 return v;
}

// Only user-provided context can establish metadata provenance; filenames and prior model assessments are excluded.
export function providedContext(body:any){return JSON.stringify({question:body.question,labels:body.scenes.map((s:any)=>typeof s.label==='string'?s.label:''),history:Array.isArray(body.history)?body.history.map((m:any)=>({role:m.role,text:m.text})):[]});}
