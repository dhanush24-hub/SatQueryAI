import json
import sqlite3
from typing import List, Optional
from app.db.session import get_db_connection
from app.schemas.image_asset import ImageAsset
from app.domain.modalities import Modality, ImageFormat
from app.core.logging import logger


class AssetRepository:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def _get_conn(self) -> sqlite3.Connection:
        return get_db_connection(self.db_path)

    def save_asset(
        self,
        asset: ImageAsset,
        internal_storage_path: str,
        internal_preview_path: Optional[str] = None
    ) -> ImageAsset:
        conn = self._get_conn()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO image_assets (
                        id, filename, storage_path, original_filename, format, modality,
                        width, height, bands, crs, resolution, bbox_json, nodata,
                        acquisition_time, sensor, metadata_json, validation_status,
                        warnings_json, preview_path
                    ) VALUES (
                        :id, :filename, :storage_path, :original_filename, :format, :modality,
                        :width, :height, :bands, :crs, :resolution, :bbox_json, :nodata,
                        :acquisition_time, :sensor, :metadata_json, :validation_status,
                        :warnings_json, :preview_path
                    )
                    ON CONFLICT(id) DO UPDATE SET
                        filename = excluded.filename,
                        storage_path = excluded.storage_path,
                        original_filename = excluded.original_filename,
                        format = excluded.format,
                        modality = excluded.modality,
                        width = excluded.width,
                        height = excluded.height,
                        bands = excluded.bands,
                        crs = excluded.crs,
                        resolution = excluded.resolution,
                        bbox_json = excluded.bbox_json,
                        nodata = excluded.nodata,
                        acquisition_time = excluded.acquisition_time,
                        sensor = excluded.sensor,
                        metadata_json = excluded.metadata_json,
                        validation_status = excluded.validation_status,
                        warnings_json = excluded.warnings_json,
                        preview_path = excluded.preview_path;
                    """,
                    {
                        "id": asset.id,
                        "filename": asset.filename,
                        "storage_path": internal_storage_path,
                        "original_filename": asset.original_filename,
                        "format": str(asset.format.value if hasattr(asset.format, 'value') else asset.format),
                        "modality": str(asset.modality.value if hasattr(asset.modality, 'value') else asset.modality),
                        "width": asset.width,
                        "height": asset.height,
                        "bands": asset.bands,
                        "crs": asset.crs,
                        "resolution": asset.resolution,
                        "bbox_json": json.dumps(asset.bbox) if asset.bbox else None,
                        "nodata": asset.nodata,
                        "acquisition_time": asset.acquisition_time,
                        "sensor": asset.sensor,
                        "metadata_json": json.dumps({
                            "raw": asset.metadata,
                            "geographic_bbox": asset.geographic_bbox,
                            "affine_transform": asset.affine_transform,
                            "driver": asset.driver,
                            "band_descriptions": asset.band_descriptions,
                            "dtypes": asset.dtypes,
                        }),
                        "validation_status": asset.validation_status,
                        "warnings_json": json.dumps(asset.warnings),
                        "preview_path": internal_preview_path,
                    }
                )
            logger.info(f"Persisted ImageAsset {asset.id} to SQLite database")
            return asset
        finally:
            conn.close()

    def get_asset(self, asset_id: str) -> Optional[ImageAsset]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM image_assets WHERE id = ?", (asset_id,)).fetchone()
            if not row:
                return None
            return self._row_to_asset(row)
        finally:
            conn.close()

    def get_internal_paths(self, asset_id: str) -> Optional[dict]:
        """Retrieve internal server paths for backend-only processing (storage & preview)."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT storage_path, preview_path FROM image_assets WHERE id = ?", (asset_id,)).fetchone()
            if not row:
                return None
            return {
                "storage_path": row["storage_path"],
                "preview_path": row["preview_path"]
            }
        finally:
            conn.close()

    def list_assets(self) -> List[ImageAsset]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM image_assets ORDER BY created_at DESC").fetchall()
            return [self._row_to_asset(r) for r in rows]
        finally:
            conn.close()

    def delete_asset(self, asset_id: str) -> bool:
        conn = self._get_conn()
        try:
            with conn:
                cursor = conn.execute("DELETE FROM image_assets WHERE id = ?", (asset_id,))
                return cursor.rowcount > 0
        finally:
            conn.close()

    def _row_to_asset(self, row: sqlite3.Row) -> ImageAsset:
        meta_dict = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else None
        warnings = json.loads(row["warnings_json"]) if row["warnings_json"] else []

        # Mask internal storage path into safe public path
        safe_storage_key = f"storage/{row['filename']}"
        has_preview = bool(row["preview_path"])
        preview_url = f"/api/uploads/{row['id']}/preview" if has_preview else None

        return ImageAsset(
            id=row["id"],
            filename=row["filename"],
            storage_path=safe_storage_key,
            original_filename=row["original_filename"],
            format=ImageFormat(row["format"]) if row["format"] in ImageFormat._value2member_map_ else ImageFormat.UNKNOWN,
            modality=Modality(row["modality"]) if row["modality"] in Modality._value2member_map_ else Modality.UNKNOWN,
            width=row["width"],
            height=row["height"],
            bands=row["bands"],
            crs=row["crs"],
            resolution=row["resolution"],
            bbox=bbox,
            geographic_bbox=meta_dict.get("geographic_bbox"),
            nodata=row["nodata"],
            acquisition_time=row["acquisition_time"],
            sensor=row["sensor"],
            affine_transform=meta_dict.get("affine_transform"),
            driver=meta_dict.get("driver"),
            band_descriptions=meta_dict.get("band_descriptions", []),
            dtypes=meta_dict.get("dtypes", []),
            preview_url=preview_url,
            has_preview=has_preview,
            metadata=meta_dict.get("raw", {}),
            validation_status=row["validation_status"],
            warnings=warnings
        )


asset_repository = AssetRepository()
