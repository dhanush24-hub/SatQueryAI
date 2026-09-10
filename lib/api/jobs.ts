/**
 * SatQuery AI — Jobs API Client
 */

import { apiClient } from './client';
import type { AnalysisStatusType } from './analysis';

export interface JobStatusDTO {
  job_id: string;
  status: AnalysisStatusType;
  progress: number;
  message?: string | null;
  result_id?: string | null;
}

export async function getJob(jobId: string): Promise<JobStatusDTO> {
  return apiClient<JobStatusDTO>(`/api/jobs/${jobId}`);
}
