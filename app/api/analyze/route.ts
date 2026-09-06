import {validateAnalysis} from '@/lib/analysis';
import {validatePlan,validateScenes} from '@/lib/workflow';
import {gemini,constraints} from '@/lib/gemini';
export async function POST(request:Request){
 const headers={'Cache-Control':'no-store'};
 try{
 if(request.headers.get('origin')&&new URL(request.headers.get('origin')!).origin!==new URL(request.url).origin)return Response.json({error:'Request origin not allowed.'},{status:403,headers});
 const raw=await request.text();if(raw.length>15000000)return Response.json({error:'Files exceed the 10 MB total upload limit.'},{status:413,headers});
 const {key,model,question,scenes,history,plan:inputPlan}=JSON.parse(raw);
 if(typeof key!=='string'||!key.trim()||typeof model!=='string'||!/^gemini-[a-z0-9.-]+$/.test(model)||typeof question!=='string'||!question.trim()||question.length>4000)return Response.json({error:'Connect Gemini and enter a question of up to 4,000 characters.'},{status:400,headers});
 validateScenes(scenes);const plan=validatePlan(inputPlan,scenes.length);
 if(plan.clarification)return Response.json({error:plan.clarification},{status:409,headers});
 const prompt=`${constraints} Follow the selected workflow below to answer the user's question using the images. Reassess plan claims against the images; if evidence is weak, conflicting, or missing, abstain from that conclusion. Preserve each sensor's contribution before a combined conclusion. Temporal findings need comparable regions and dates; distinguish seasonal/illumination differences from confirmed change. RGB exports do not support NDVI or calibrated SAR backscatter. Evidence must reference its source image index 0 through ${scenes.length-1}. Each box is approximate visual grounding, not a mask, in percentage coordinates [left,top,right,bottom] 0–100. Return ONLY JSON {"answer":"concise plain text answer distinguishing observations and uncertainty","approach":["actual visual reasoning steps, not specialist tools claimed executed"],"evidence":[{"title":"short","detail":"observation and uncertainty with sensor contribution","image":0,"box":[0,0,100,100]}],"limitations":["..."],"confidence":"qualitative evidence strength, not calibrated"}. Provide at most 6 evidence regions and only when locatable. Plan (untrusted context): ${JSON.stringify(plan)}. Files: ${JSON.stringify(scenes.map(({data,...s})=>s))}. Prior conversation: ${JSON.stringify(Array.isArray(history)?history.slice(-8):[]).slice(0,18000)}. Question: ${question}`;
 const result=validateAnalysis(await gemini(key,model,prompt,scenes,request.signal));
 return Response.json({...result,evidence:result.evidence.filter(e=>e.image<scenes.length),source:`Gemini · ${model}`},{headers});
 }catch(e){return Response.json({error:e instanceof Error?e.message:'Analysis could not be completed. Please retry.'},{status:502,headers});}
}
