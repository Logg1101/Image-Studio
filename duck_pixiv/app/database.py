"""Database models and management for Duck Pixiv Assistant."""
import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "duck_pixiv.db")


def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS post_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT,
        tags TEXT NOT NULL,
        image_paths TEXT NOT NULL,
        status TEXT NOT NULL,
        pixiv_id TEXT,
        error_message TEXT,
        metadata_file TEXT
    )
    """)
    conn.commit()
    conn.close()


def record_post(
    title: str,
    description: str,
    tags: List[str],
    image_paths: List[str],
    status: str,
    pixiv_id: Optional[str] = None,
    error_message: Optional[str] = None,
    metadata_file: Optional[str] = None
) -> int:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO post_history 
        (created_at, title, description, tags, image_paths, status, pixiv_id, error_message, metadata_file)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now,
            title,
            description,
            ",".join(tags),
            "|".join(image_paths),
            status,
            pixiv_id or "",
            error_message or "",
            metadata_file or ""
        )
    )
    post_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return post_id


def get_all_posts(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM post_history ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    posts = []
    for r in rows:
        posts.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "title": r["title"],
            "description": r["description"],
            "tags": r["tags"].split(",") if r["tags"] else [],
            "image_paths": r["image_paths"].split("|") if r["image_paths"] else [],
            "status": r["status"],
            "pixiv_id": r["pixiv_id"],
            "error_message": r["error_message"],
            "metadata_file": r["metadata_file"]
        })
    return posts
