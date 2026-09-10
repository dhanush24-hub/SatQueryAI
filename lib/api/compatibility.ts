/**
 * SatQuery AI — Pair Compatibility & Alignment API Client
 */

import { apiClient } from './client';
import { ImageAssetResponse } from './uploads';

export type PairType = 'TEMPORAL' | 'OPTICAL_SAR';

export interface PairCompatibilityRequest {
  asset_id_a: string;
  asset_id_b: string;
  pair_type?: PairType;
}

export interface PairCompatibilityReport {
  compatible: boolean;
  pair_type: PairType;
  asset_id_a: string;
  asset_id_b: string;
  crs_a?: string | null;
  crs_b?: string | null;
  crs_match: boolean;
  spatial_overlap_percentage: number;
  has_spatial_overlap: boolean;
  overlap_geographic_bbox?: number[] | null;
  resolution_ratio: number;
  requires_reprojection: boolean;
  requires_resampling: boolean;
  is_coregistered: boolean;
  modality_compatible: boolean;
  acquisition_dates?: (string | null)[] | null;
  warnings: string[];
  recommendations: string[];
}

export interface AlignmentRequest {
  source_asset_id: string;
  reference_asset_id: string;
  resampling_method?: 'nearest' | 'bilinear' | 'cubic';
}

export async function checkPairCompatibility(
  request: PairCompatibilityRequest
): Promise<PairCompatibilityReport> {
  return apiClient<PairCompatibilityReport>('/api/compatibility/check', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function alignImageryPair(
  request: AlignmentRequest
): Promise<ImageAssetResponse> {
  return apiClient<ImageAssetResponse>('/api/compatibility/align', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}
