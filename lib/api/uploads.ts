/**
 * SatQuery AI — Uploads API Client
 */

import { apiClient } from './client';

export type ModalityType = 'OPTICAL' | 'MULTISPECTRAL' | 'SAR' | 'UNKNOWN';
export type ImageFormatType = 'GEOTIFF' | 'TIFF' | 'PNG' | 'JPEG' | 'WEBP' | 'UNKNOWN';

export interface ImageAssetResponse {
  id: string;
  filename: string;
  storage_path: string;
  original_filename: string;
  format: ImageFormatType;
  modality: ModalityType;
  width?: number | null;
  height?: number | null;
  bands?: number | null;
  crs?: string | null;
  resolution?: number | null;
  bbox?: number[] | null;
  nodata?: number | null;
  acquisition_time?: string | null;
  sensor?: string | null;
  preview_url?: string | null;
  metadata: Record<string, any>;
  validation_status: string;
  warnings: string[];
}

export interface UploadOptions {
  benchmarkMode?: boolean;
  modality?: ModalityType;
}

export async function uploadImagery(files: File[], options?: UploadOptions): Promise<ImageAssetResponse[]> {
  if (!files || files.length === 0) {
    throw new Error('At least one file must be provided for upload.');
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  if (options?.benchmarkMode) {
    formData.append('benchmark_mode', 'true');
  }
  if (options?.modality) {
    formData.append('modality', options.modality);
  }

  return apiClient<ImageAssetResponse[]>('/api/uploads', {
    method: 'POST',
    body: formData,
  });
}

export async function getAssetMetadata(assetId: string): Promise<ImageAssetResponse> {
  return apiClient<ImageAssetResponse>(`/api/uploads/${assetId}`, {
    method: 'GET',
  });
}

export function getAssetPreviewUrl(assetId: string): string {
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
  return `${baseUrl}/api/uploads/${assetId}/preview`;
}
