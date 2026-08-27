"""
scripts/seed_curriculum.py
--------------------------
Seeds the Machine Learning curriculum knowledge graph into PostgreSQL or SQLite.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from ai_tutor.concept_graph import ML_CONCEPTS, ML_EDGES


def seed_sqlite(db_path: str = "ai_tutor.db"):
    import sqlite3
    print(f"[*] Seeding curriculum graph into SQLite database at '{db_path}'...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            concept_id TEXT PRIMARY KEY,
            domain TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS concept_prerequisites (
            concept_id TEXT,
            prerequisite_id TEXT,
            weight REAL DEFAULT 1.0,
            PRIMARY KEY (concept_id, prerequisite_id)
        );
    """)

    for c in ML_CONCEPTS:
        cur.execute("""
            INSERT OR REPLACE INTO concepts (concept_id, domain, name, description)
            VALUES (?, ?, ?, ?)
        """, (c.concept_id, c.domain, c.name, c.description or ""))

    for e in ML_EDGES:
        cur.execute("""
            INSERT OR REPLACE INTO concept_prerequisites (concept_id, prerequisite_id, weight)
            VALUES (?, ?, ?)
        """, (e.concept_id, e.prerequisite_id, e.weight))

    conn.commit()
    conn.close()
    print(f"[+] Successfully seeded {len(ML_CONCEPTS)} concepts and {len(ML_EDGES)} prerequisite edges.")


if __name__ == "__main__":
    db_file = sys.argv[1] if len(sys.argv) > 1 else "ai_tutor.db"
    seed_sqlite(db_file)
