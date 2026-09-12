import sqlite3
import os
import glob
import json
import time
import datetime
from pathlib import Path
from PIL import Image
import config.paths as paths

class HistoryDB:
    def __init__(self):
        self.db_path = paths.PROJECT_ROOT / "history" / "generations.db"
        self._last_sync_time = 0.0
        self._init_db()
        
    def _init_db(self):
        """Creates the generations table if it does not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    architecture TEXT,
                    variant TEXT,
                    prompt TEXT,
                    negative_prompt TEXT,
                    seed INTEGER,
                    steps INTEGER,
                    guidance_scale REAL,
                    width INTEGER,
                    height INTEGER,
                    image_path TEXT,
                    generation_time_ms REAL,
                    vram_peak_gb REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Cleanup any existing duplicate paths before creating unique index
            conn.execute("""
                DELETE FROM generations 
                WHERE id NOT IN (
                    SELECT MIN(id) FROM generations 
                    GROUP BY LOWER(REPLACE(image_path, '/', '\\'))
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_generations_timestamp ON generations(timestamp DESC)")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_generations_image_path ON generations(image_path)")
    
    def insert_record(self, record: dict) -> int:
        """Inserts a new generation record into the database, ignoring duplicates."""
        rec = record.copy()
        if "image_path" in rec and rec["image_path"]:
            rec["image_path"] = os.path.normpath(os.path.abspath(rec["image_path"]))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO generations (
                    model_id, architecture, variant, prompt, negative_prompt,
                    seed, steps, guidance_scale, width, height,
                    image_path, generation_time_ms, vram_peak_gb
                ) VALUES (
                    :model_id, :architecture, :variant, :prompt, :negative_prompt,
                    :seed, :steps, :guidance_scale, :width, :height,
                    :image_path, :generation_time_ms, :vram_peak_gb
                )
            """, rec)
            conn.commit()
            return cursor.lastrowid or 0

    def sync_from_disk(self, force: bool = False) -> dict:
        """
        Scans outputs/ directory for any PNG files not yet tracked in the database,
        parses embedded parameters metadata, and inserts them chronologically.
        """
        now = time.time()
        if not force and (now - self._last_sync_time < 5.0):
            return {"synced": 0, "total": self.get_total_count()}

        self._last_sync_time = now
        outputs_dir = paths.OUTPUTS_DIR
        if not outputs_dir.exists():
            return {"synced": 0, "total": 0}

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            existing_rows = conn.execute("SELECT image_path FROM generations").fetchall()
            existing_paths = {
                os.path.normpath(r[0]).lower()
                for r in existing_rows
                if r[0]
            }

            def _is_valid_generation(p_str: str) -> bool:
                p_obj = Path(p_str)
                for part in p_obj.parts:
                    pl = part.lower()
                    if pl.startswith(("test_", ".", "metadata", "enhancer_debug")):
                        return False
                filename = p_obj.name.lower()
                if any(x in filename for x in ["_depth_map", "_normal_map", "_diffuse", "prep_map"]):
                    return False
                return True

            all_pngs = glob.glob(str(outputs_dir / "**" / "*.png"), recursive=True)
            missing = [
                os.path.normpath(os.path.abspath(p))
                for p in all_pngs
                if _is_valid_generation(p) and os.path.normpath(os.path.abspath(p)).lower() not in existing_paths
            ]

            if not missing:
                return {"synced": 0, "total": len(existing_paths)}

            new_records = []
            for p in missing:
                try:
                    mtime = os.path.getmtime(p)
                    dt_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    with Image.open(p) as img:
                        w, h = img.size
                        raw_params = img.info.get("parameters")
                        params = {}
                        if raw_params:
                            try:
                                params = json.loads(raw_params)
                            except Exception:
                                pass

                        prompt = params.get("prompt") or Path(p).stem
                        model_id = params.get("model") or "sdxl"
                        architecture = params.get("architecture") or "sdxl"
                        variant = params.get("variant") or "base"
                        neg_prompt = params.get("negative_prompt") or ""
                        seed = int(params.get("seed") or 0)
                        steps = int(params.get("steps") or 28)
                        guidance_scale = float(params.get("guidance_scale") or 7.0)
                        width = int(params.get("width") or w)
                        height = int(params.get("height") or h)
                        time_ms = float(params.get("generation_time_ms") or 0.0)
                        vram_gb = float(params.get("vram_peak_gb") or 0.0)

                        new_records.append((
                            model_id,
                            architecture,
                            variant,
                            prompt,
                            neg_prompt,
                            seed,
                            steps,
                            guidance_scale,
                            width,
                            height,
                            p,
                            time_ms,
                            vram_gb,
                            dt_str,
                        ))
                except Exception as ex:
                    print(f"[HistoryDB] Skipping corrupt/unreadable file {p}: {ex}")

            if new_records:
                # Sort ascending by timestamp so row IDs align with time
                new_records.sort(key=lambda r: r[13])
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR IGNORE INTO generations (
                        model_id, architecture, variant, prompt, negative_prompt,
                        seed, steps, guidance_scale, width, height,
                        image_path, generation_time_ms, vram_peak_gb, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, new_records)
                conn.commit()

        total = self.get_total_count()
        return {"synced": len(new_records), "total": total}

    def get_records(self, limit: int = 100, offset: int = 0, search: str = "", auto_sync: bool = True) -> list:
        """Queries generation records with optional search and pagination, newest first."""
        if auto_sync:
            try:
                self.sync_from_disk(force=False)
            except Exception as e:
                print(f"[HistoryDB] Auto-sync notice: {e}")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if search and search.strip():
                query = f"%{search.strip()}%"
                cursor.execute("""
                    SELECT * FROM generations
                    WHERE prompt LIKE ? OR model_id LIKE ? OR architecture LIKE ?
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ? OFFSET ?
                """, (query, query, query, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM generations
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_total_count(self, search: str = "") -> int:
        """Returns total record count matching search filter."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if search and search.strip():
                query = f"%{search.strip()}%"
                cursor.execute("""
                    SELECT COUNT(*) FROM generations
                    WHERE prompt LIKE ? OR model_id LIKE ? OR architecture LIKE ?
                """, (query, query, query))
            else:
                cursor.execute("SELECT COUNT(*) FROM generations")
            return cursor.fetchone()[0]

    def delete_record(self, record_id: int, delete_file: bool = True) -> bool:
        """Deletes a generation record by ID and optionally removes files from disk."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if delete_file:
                cursor.execute("SELECT image_path FROM generations WHERE id = ?", (record_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    img_path = Path(row[0])
                    try:
                        if img_path.exists():
                            img_path.unlink()
                    except OSError as ex:
                        print(f"[HistoryDB] Could not delete image file {img_path}: {ex}")
                    try:
                        meta_file = paths.METADATA_DIR / f"{img_path.stem}.json"
                        if meta_file.exists():
                            meta_file.unlink()
                    except OSError as ex:
                        print(f"[HistoryDB] Could not delete metadata file {meta_file}: {ex}")

            cursor.execute("DELETE FROM generations WHERE id = ?", (record_id,))
            conn.commit()
            return cursor.rowcount > 0