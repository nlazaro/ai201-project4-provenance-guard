"""db.py — SQLite storage for content status and the structured audit log."""
import sqlite3, json, os
from datetime import datetime, timezone

DB_PATH = os.environ.get("PG_DB_PATH", os.path.join(os.path.dirname(__file__), "provenance.db"))


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS content (
                content_id   TEXT PRIMARY KEY,
                creator_id   TEXT,
                text_excerpt TEXT,
                attribution  TEXT,
                confidence   REAL,
                ai_likelihood REAL,
                status       TEXT,
                created_at   TEXT
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id   TEXT,
                creator_id   TEXT,
                timestamp    TEXT,
                event_type   TEXT,
                attribution  TEXT,
                confidence   REAL,
                ai_likelihood REAL,
                llm_score    REAL,
                stylometric_score REAL,
                lexical_score REAL,
                label_variant TEXT,
                status       TEXT,
                appeal_reasoning TEXT
            );
            """
        )


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def create_content(content_id, creator_id, text, result):
    sig = result["signals"]
    with _conn() as c:
        c.execute(
            "INSERT INTO content VALUES (?,?,?,?,?,?,?,?)",
            (content_id, creator_id, text[:200], result["attribution"],
             result["confidence"], result["ai_likelihood"], "classified", _now()),
        )


def log_event(content_id, creator_id, event_type, result=None, status=None, appeal_reasoning=None):
    sig = (result or {}).get("signals", {})
    with _conn() as c:
        c.execute(
            """INSERT INTO audit_log
               (content_id, creator_id, timestamp, event_type, attribution, confidence,
                ai_likelihood, llm_score, stylometric_score, lexical_score, label_variant,
                status, appeal_reasoning)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (content_id, creator_id, _now(), event_type,
             (result or {}).get("attribution"), (result or {}).get("confidence"),
             (result or {}).get("ai_likelihood"), sig.get("llm_score"),
             sig.get("stylometric_score"), sig.get("lexical_score"),
             (result or {}).get("label", {}).get("variant"), status, appeal_reasoning),
        )


def update_status(content_id, status):
    with _conn() as c:
        cur = c.execute("UPDATE content SET status=? WHERE content_id=?", (status, content_id))
        return cur.rowcount > 0


def get_content(content_id):
    with _conn() as c:
        row = c.execute("SELECT * FROM content WHERE content_id=?", (content_id,)).fetchone()
        return dict(row) if row else None


def get_log(limit=50):
    with _conn() as c:
        rows = c.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_analytics():
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM content").fetchone()[0]
        by_attr = {r["attribution"]: r["n"] for r in c.execute(
            "SELECT attribution, COUNT(*) n FROM content GROUP BY attribution")}
        appeals = c.execute(
            "SELECT COUNT(*) FROM audit_log WHERE event_type='appeal'").fetchone()[0]
        avg_conf = c.execute("SELECT AVG(confidence) FROM content").fetchone()[0]
        under_review = c.execute(
            "SELECT COUNT(*) FROM content WHERE status='under_review'").fetchone()[0]
    return {
        "total_submissions": total,
        "by_attribution": by_attr,
        "appeals_filed": appeals,
        "appeal_rate": round(appeals / total, 3) if total else 0.0,
        "content_under_review": under_review,
        "avg_confidence": round(avg_conf, 3) if avg_conf is not None else None,
    }
