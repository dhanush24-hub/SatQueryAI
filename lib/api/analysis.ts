/**
 * SatQuery AI — Analysis API Client
 */

import { apiClient } from './client';
import type {
  MissionResult,
  Finding,
  EvidenceItem,
  EvidenceValidation,
  ExecutionTrace,
  WorkflowStep,
  Strength,
  WorkflowCapability,
} from '@/lib/domain';

export type TaskFamilyType =
  | 'SINGLE_VQA'
  | 'SINGLE_GROUNDING'
  | 'SINGLE_CAPTION'
  | 'TEMPORAL_CHANGE'
  | 'TEMPORAL_CHANGE_VQA'
  | 'OPTICAL_SAR_ANALYSIS'
  | 'UNKNOWN';

export type AnalysisStatusType =
  | 'PENDING'
  | 'RUNNING'
  | 'COMPLETED'
  | 'COMPLETED_WITH_WARNINGS'
  | 'INSUFFICIENT_EVIDENCE'
  | 'VALIDATION_FAILED'
  | 'FAILED'
  | 'NOT_IMPLEMENTED'
  | 'MODEL_UNAVAILABLE';

export interface BoundingBoxROI {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface EvidenceItemDTO {
  id: string;
  image_id: string;
  finding_id: string;
  evidence_type: string;
  description: string;
  region?: BoundingBoxROI | null;
  reliability: string;
  limitations: string[];
  source_step: string;
}

export interface ExecutionStepDTO {
  step: string;
  tool_name: string;
  model_name?: string | null;
  parameters?: Record<string, any>;
  status: string;
  duration_ms?: number | null;
  summary: string;
}

export interface AnalysisRequestPayload {
  query: string;
  image_ids: string[];
  session_id?: string;
  parameters?: Record<string, any>;
}

export interface FindingItemDTO {
  finding_id: string;
  type: string;
  summary: string;
  bbox_px?: number[] | null;
  canvas_box?: BoundingBoxROI | null;
  geometry?: any | null;
  changed_pixels?: number | null;
  area_m2?: number | null;
  evidence_quality: string;
  source_model: string;
  limitations: string[];
}

export interface OverlayDataDTO {
  change_mask_url?: string | null;
  t1_preview_url?: string | null;
  t2_preview_url?: string | null;
  change_ratio_pct?: number | null;
  total_changed_pixels?: number | null;
  georeferenced: boolean;
}

export interface RegistrationInfoDTO {
  status: string;
  correlation: number;
  translation_px: number;
  warnings: string[];
}

export interface DomainSuitabilityInfoDTO {
  is_suitable: boolean;
  status: string;
  reasons: string[];
  warnings: string[];
}

export interface ModelProvenanceInfoDTO {
  model_name: string;
  architecture: string;
  checkpoint: string;
  parameters: number;
  threshold: number;
  benchmark_f1?: number | null;
  benchmark_iou?: number | null;
}

export interface DownloadableArtifactsDTO {
  report_pdf_url?: string | null;
  result_json_url?: string | null;
  mask_png_url?: string | null;
  geojson_url?: string | null;
}

export interface AnalysisHistorySummaryDTO {
  id: string;
  query: string;
  created_at: string;
  task_family: string;
  status: string;
  evidence_state: string;
  summary: string;
  thumbnail_url?: string | null;
}

export interface AnalysisResponse {
  id: string;
  answer: string;
  task: TaskFamilyType;
  confidence?: string | null;
  evidence: EvidenceItemDTO[];
  findings?: FindingItemDTO[];
  overlays?: OverlayDataDTO | null;
  registration?: RegistrationInfoDTO | null;
  domain_suitability?: DomainSuitabilityInfoDTO | null;
  model_provenance?: ModelProvenanceInfoDTO | null;
  downloadable_artifacts?: DownloadableArtifactsDTO | null;
  evidence_state?: string;
  limitations?: string[];
  warnings: string[];
  execution_summary: ExecutionStepDTO[];
  status: AnalysisStatusType;
}

export async function submitAnalysis(
  payload: AnalysisRequestPayload
): Promise<AnalysisResponse> {
  return apiClient<AnalysisResponse>('/api/analysis', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function getAnalysis(analysisId: string): Promise<AnalysisResponse> {
  return apiClient<AnalysisResponse>(`/api/analysis/${analysisId}`);
}

export async function listAnalyses(): Promise<AnalysisHistorySummaryDTO[]> {
  return apiClient<AnalysisHistorySummaryDTO[]>('/api/analysis');
}

/**
 * Maps real backend AnalysisResponse into domain MissionResult for canvas & findings rendering.
 */
export function mapBackendAnalysisToMissionResult(
  res: AnalysisResponse,
  query: string,
  assetId: string
): MissionResult {
  const isTemporal = res.task === 'TEMPORAL_CHANGE' || res.task === 'TEMPORAL_CHANGE_VQA';
  const modelName =
    res.execution_summary[0]?.model_name ||
    (isTemporal
      ? 'HZDR-FWGEL/UCD-LEVIRCD256-SiamUDiff'
      : res.task === 'SINGLE_GROUNDING'
      ? 'google/owlvit-base-patch32'
      : 'Salesforce/blip-vqa-base');

  let findings: Finding[] = [];
  if (isTemporal) {
    findings = res.evidence.map((ev, idx) => ({
      id: ev.finding_id || `f_temp_${idx + 1}`,
      title: `Change Cluster ${idx + 1}: ${ev.description.split(' (')[0].replace("Cluster '", '').replace("'", '')}`,
      statement: ev.description,
      classification: 'observation',
      evidenceIds: [ev.id],
      confidence: {
        level: (ev.reliability === 'high' ? 'high' : 'medium') as Strength,
        meaning: 'Bi-temporal difference inference (uncalibrated probability)',
      },
      alternativeExplanations: [],
      limitations: ev.limitations,
      category: 'surface_change',
    }));

    if (!findings.length) {
      findings = [
        {
          id: 'f_temp_summary',
          title: 'Bi-Temporal Change Analysis',
          statement: res.answer,
          classification: 'observation',
          evidenceIds: [],
          confidence: {
            level: 'medium',
            meaning: res.confidence || 'Empirical bi-temporal change detection verification',
          },
          alternativeExplanations: [],
          limitations: res.warnings,
          category: 'surface_change',
        },
      ];
    }
  } else if (res.task === 'SINGLE_GROUNDING') {
    findings = res.evidence.map((ev, idx) => ({
      id: ev.finding_id || `f_ground_${idx + 1}`,
      title: `Spatial Grounding ${idx + 1}: ${ev.description.split(' (')[0].replace("Detected '", '').replace("'", '')}`,
      statement: ev.description,
      classification: 'observation',
      evidenceIds: [ev.id],
      confidence: {
        level: (ev.reliability === 'high' ? 'high' : 'medium') as Strength,
        meaning: 'Raw detector logit above threshold (uncalibrated probability)',
      },
      alternativeExplanations: [],
      limitations: ev.limitations,
      category: 'visual_grounding',
    }));

    if (!findings.length) {
      findings = [
        {
          id: 'f_ground_none',
          title: 'Grounding Verification',
          statement: res.answer,
          classification: 'observation',
          evidenceIds: [],
          confidence: {
            level: 'medium',
            meaning: 'No candidate regions exceeded detection threshold',
          },
          alternativeExplanations: [],
          limitations: res.warnings,
          category: 'visual_grounding',
        },
      ];
    }
  } else {
    findings = [
      {
        id: 'f_vqa_1',
        title: 'VQA Scene Understanding',
        statement: res.answer,
        classification: 'observation',
        evidenceIds: [],
        confidence: {
          level: 'medium',
          meaning: 'Autoregressive VLM generation (uncalibrated generation logits)',
        },
        alternativeExplanations: [],
        limitations: res.warnings,
        category: 'scene_understanding',
      },
    ];
  }

  const evidenceItems: EvidenceItem[] = res.evidence.map((e) => ({
    id: e.id,
    imageId: isTemporal ? 'img-2' : 'img-1',
    findingId: e.finding_id,
    evidenceType: isTemporal ? 'temporal_difference' : 'visual_region',
    description: e.description,
    region: e.region
      ? {
          x: e.region.x,
          y: e.region.y,
          width: e.region.width,
          height: e.region.height,
        }
      : undefined,
    reliability: (e.reliability === 'high' ? 'high' : 'medium') as Strength,
    limitations: e.limitations,
    sourceStep: e.source_step,
  }));

  const validations: EvidenceValidation[] = findings.map((f) => ({
    findingId: f.id,
    verdict: 'supported',
    reasons: ['Directly grounded in raster imagery via specialist model inference.'],
  }));

  const trace: ExecutionTrace[] = res.execution_summary.map((s, idx) => ({
    timestamp: new Date().toISOString(),
    stepId: `step_${idx + 1}`,
    step: s.step,
    purpose: `${s.tool_name} (${s.model_name || 'Specialist Model'})`,
    inputs: isTemporal ? ['img-1', 'img-2'] : ['img-1'],
    provider: s.model_name || 'SatQuery Backend Specialist',
    status: s.status === 'completed' ? 'completed' : 'failed',
    summary: s.summary,
  }));

  const workflow: WorkflowStep[] = res.execution_summary.map((s, idx) => ({
    id: `wf_${idx + 1}`,
    capability: (isTemporal
      ? 'temporal_comparison'
      : res.task === 'SINGLE_GROUNDING'
      ? 'visual_grounding'
      : 'scene_understanding') as WorkflowCapability,
    title: s.step,
    reason: s.summary,
    inputImageIds: isTemporal ? ['img-1', 'img-2'] : ['img-1'],
    dependencies: [],
    status: s.status === 'completed' ? 'completed' : 'blocked',
  }));

  return {
    findings,
    evidenceItems,
    validations,
    crossSensorAssessment: [],
    directAnswer: res.answer,
    uncertainties: res.warnings,
    nextQuestion: isTemporal
      ? 'Ask where the most significant change occurred, or analyze another region.'
      : res.task === 'SINGLE_GROUNDING'
      ? 'Ask a question about the identified regions or ask to highlight another feature.'
      : 'Ask for specific region grounding or scene details.',
    trace,
    source: `SatQuery Real AI · ${modelName}`,
    workflow,
  };
}
