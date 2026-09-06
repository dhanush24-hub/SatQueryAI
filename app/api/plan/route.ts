import {gemini,constraints} from '@/lib/gemini';
import {validatePlan} from '@/lib/workflow';
import {plannerPrompt} from '@/lib/prompts/planner';
import {readInput,providedContext} from '@/lib/api-input';
export async function POST(request:Request){
 try{const {key,model,question,scenes,history,mission}=await readInput(request);
 const metadata=scenes.map(({data,...s}:any)=>s);const context=JSON.stringify({question,metadata,history,mission}).slice(0,160000);
 const raw=await gemini(key,model,`${constraints}\n${plannerPrompt}\nMission context: ${context}`,scenes,request.signal);
 return Response.json(validatePlan(raw,scenes.length,providedContext({question,scenes,history})),{headers:{'Cache-Control':'no-store'}});
 }catch(e){return Response.json({error:e instanceof Error?e.message:'Planning failed.'},{status:400,headers:{'Cache-Control':'no-store'}});}
}
