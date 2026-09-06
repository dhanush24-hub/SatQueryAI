import {readInput,providedContext} from '@/lib/api-input';
import {validatePlan,assertExecutable} from '@/lib/workflow';
import {reviewPlan} from '@/lib/demos';
import {executePlan} from '@/lib/executor';
export async function POST(request:Request){
 try{
 const body=await readInput(request);const context=JSON.stringify({question:body.question,history:body.history,metadata:body.scenes.map(({data,...s}:any)=>s),mission:body.mission}).slice(0,160000);
 let plan=validatePlan(body.plan,body.scenes.length,providedContext(body));if(body.reviewExisting&&body.mission?.previousResult)plan=reviewPlan(plan);assertExecutable(plan);
 const abort=new AbortController();const signal=AbortSignal.any([request.signal,abort.signal]);
 const stream=new ReadableStream({start(controller){const encoder=new TextEncoder();const emit=(event:unknown)=>{if(!signal.aborted)controller.enqueue(encoder.encode(JSON.stringify(event)+'\n'));};
 void executePlan({key:body.key,model:body.model,question:body.question,scenes:body.scenes,plan,mission:body.mission,signal},emit).catch(e=>emit({type:'error',error:e instanceof Error?e.message:'Analysis failed.'})).finally(()=>{try{controller.close();}catch{}});
 },cancel(){abort.abort();}});
 return new Response(stream,{headers:{'Content-Type':'application/x-ndjson','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}});
 }catch(e){return Response.json({error:e instanceof Error?e.message:'Analysis could not start.'},{status:409,headers:{'Cache-Control':'no-store'}});}
}
