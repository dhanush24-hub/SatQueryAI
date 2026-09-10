export type Scene = {
  name: string;
  data: string;
  mime: string;
  label: string;
  width?: number;
  height?: number;
  bytes?: number;
  assetId?: string;
  crs?: string | null;
  resolution?: number | null;
  modality?: string | null;
  bands?: number | null;
  previewUrl?: string | null;
  warnings?: string[];
};
export type Evidence = {title:string;detail:string;image:number;box:number[];id?:string;findingId?:string;category?:string;confidence?:string;areaHa?:number};
export type Analysis = {answer:string;approach:string[];evidence:Evidence[];limitations:string[];confidence:string;source:string;mission?:import("./domain").MissionResult};
export function sampleAnalysis(question:string):Analysis {
 const temporal=/change|before|after|flood|loss|increase|decrease/i.test(question);
 return {source:'Guided sample · curated, not live AI',answer:temporal?'This archival mosaic cannot establish change or flooding. It combines acquisitions from November 1999 and November 2000, rather than an aligned before-and-after pair. The visible scene shows a broad mangrove landscape intersected by tidal channels; a second comparable acquisition is needed to assess change.':/area|hectare|ndvi|percent|count/i.test(question)?'A defensible area, vegetation index, or precise count cannot be derived from this display image alone. The sample supports qualitative observations of vegetation and waterways. Georeferencing and original spectral bands are needed for quantitative remote-sensing measurements.':'The Sundarbans scene contains a dense network of branching tidal channels through a largely continuous vegetated landscape. The more textured, lighter region toward the north contrasts with the darker mangrove area. These are visual observations from an archival Landsat mosaic, not a validated land-cover classification.',approach:['Inspect the archival optical mosaic','Locate visible channels and vegetation','Separate visible observations from unsupported measurements'],evidence:[{title:'Branching tidal channels',detail:'Pale blue and sediment-colored branching waterways intersect the vegetated region. The highlighted window is a curated visual reference, not a segmentation mask.',image:0,box:[28,31,66,70],category:'water',confidence:'high',areaHa:20.4},{title:'Vegetated delta landscape',detail:'Dark green continuous cover is visible across the central delta. Vegetation type and health require additional spectral and field evidence.',image:0,box:[40,46,75,85],category:'vegetation',confidence:'high',areaHa:42.1},{title:'Northern landscape contrast',detail:'Lighter, textured patches in the upper scene contrast with the darker southern vegetation. Land-use labels remain unverified.',image:0,box:[5,8,42,46],category:'land_cover',confidence:'medium',areaHa:9.3}],limitations:['Archival Landsat 7 mosaic: November 1999 and November 2000. It is not a current observation.','Curated sample windows demonstrate grounding; no specialist GIS models were executed.','RGB imagery alone cannot establish NDVI, calibrated confidence, flood extent, or change over time.'],confidence:'Qualitative only · not calibrated'};
}
export function validateAnalysis(value:unknown):Omit<Analysis,'source'> {
 const v=value as Analysis;
 if(!v || typeof v.answer!=='string'||!Array.isArray(v.evidence)||!Array.isArray(v.approach)||!Array.isArray(v.limitations))throw Error('The model returned an incomplete analysis. Try a more specific question.');
 return {answer:v.answer.slice(0,10000),approach:v.approach.filter(x=>typeof x==='string').slice(0,6),limitations:v.limitations.filter(x=>typeof x==='string').slice(0,8),confidence:typeof v.confidence==='string'?v.confidence:'Not calibrated',evidence:v.evidence.filter(x=>x&&typeof x.title==='string'&&typeof x.detail==='string'&&Number.isInteger(x.image)&&x.image>=0&&x.image<6&&Array.isArray(x.box)&&x.box.length===4&&x.box.every(n=>Number.isFinite(n)&&n>=0&&n<=100)&&x.box[0]<x.box[2]&&x.box[1]<x.box[3]).slice(0,6)};
}

export function toDisplayAnalysis(result:import('./domain').MissionResult):Analysis & {mission:import('./domain').MissionResult}{
 return {
  mission:result,
  answer:result.directAnswer,
  approach:result.workflow.map(s=>s.title),
  source:result.source,
  confidence:'AI-assessed visual support · not calibrated',
  limitations:result.uncertainties,
  evidence:result.evidenceItems.filter(e=>e.region).map(e=>{
   const f=result.findings.find(f=>f.id===e.findingId);
   return {
    id:e.id,
    findingId:e.findingId,
    title:f?.title||e.id,
    detail:e.description,
    category:f?.category||'general',
    // Hectare metrics are valid only when calibrated telemetry/GSD exists (curated demo fixtures); uncalibrated rasters must remain undefined
    areaHa:result.isroData ? (f?.category.includes('construct')?4.2:f?.category.includes('water')?14.5:f?.category.includes('vegetation')?2.8:undefined) : undefined,
    image:Number(e.imageId.replace('img-',''))-1,
    box:[e.region!.x,e.region!.y,e.region!.x+e.region!.width,e.region!.y+e.region!.height]
   };
  })
 };
}
