import os
import sqlite3
from pathlib import Path
from app.core.config import settings
from app.core.logging import logger


_INITIALIZED_PATHS = set()

def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    path = db_path or settings.DATABASE_PATH
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    
    if path not in _INITIALIZED_PATHS:
        _init_tables(conn, path)
        _INITIALIZED_PATHS.add(path)
        
    return conn

def _init_tables(conn: sqlite3.Connection, path: str):
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS image_assets (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            storage_path TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            format TEXT NOT NULL,
            modality TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            bands INTEGER,
            crs TEXT,
            resolution REAL,
            bbox_json TEXT,
            nodata REAL,
            acquisition_time TEXT,
            sensor TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            validation_status TEXT NOT NULL DEFAULT 'valid',
            warnings_json TEXT NOT NULL DEFAULT '[]',
            preview_path TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_format ON image_assets(format);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_modality ON image_assets(modality);")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            task TEXT NOT NULL,
            status TEXT NOT NULL,
            answer TEXT NOT NULL,
            confidence TEXT,
            evidence_state TEXT NOT NULL,
            image_ids_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            report_pdf_path TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_created ON analysis_results(created_at);")
    logger.info(f"Initialized SQLite database tables at: {path}")


def init_db(db_path: str = None) -> None:
    conn = get_db_connection(db_path)
    try:
        _init_tables(conn, db_path or settings.DATABASE_PATH)
    finally:
        conn.close()
