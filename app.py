"""
app.py — Provenance Guard API.

Endpoints
  POST /submit     {text, creator_id}            -> classify, label, log, return result
  POST /appeal     {content_id, creator_reasoning} -> status=under_review, log, confirm
  GET  /log        ?limit=N                       -> recent audit-log entries
  GET  /analytics  (stretch)                      -> detection patterns + appeal rate
  GET  /health                                    -> liveness
"""
import os, uuid
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

import db
import detection

load_dotenv()
app = Flask(__name__)
db.init_db()

# Rate limiting — see README for the reasoning behind these specific numbers.
limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    creator_id = data.get("creator_id")
    if not text or not creator_id:
        return jsonify({"error": "both 'text' and 'creator_id' are required"}), 400

    result = detection.analyze(text)
    content_id = str(uuid.uuid4())

    db.create_content(content_id, creator_id, text, result)
    db.log_event(content_id, creator_id, "submission", result=result, status="classified")

    return jsonify({
        "content_id": content_id,
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "ai_likelihood": result["ai_likelihood"],
        "signals": result["signals"],
        "label": result["label"],
        "status": "classified",
    }), 200


@app.route("/appeal", methods=["POST"])
@limiter.limit("20 per hour")
def appeal():
    data = request.get_json(silent=True) or {}
    content_id = data.get("content_id")
    reasoning = (data.get("creator_reasoning") or "").strip()
    if not content_id or not reasoning:
        return jsonify({"error": "both 'content_id' and 'creator_reasoning' are required"}), 400

    content = db.get_content(content_id)
    if not content:
        return jsonify({"error": f"no content found with id {content_id}"}), 404

    db.update_status(content_id, "under_review")
    # Log the appeal alongside the original decision (carried from the content row).
    original = {
        "attribution": content["attribution"],
        "confidence": content["confidence"],
        "ai_likelihood": content["ai_likelihood"],
        "signals": {}, "label": {},
    }
    db.log_event(content_id, content["creator_id"], "appeal",
                 result=original, status="under_review", appeal_reasoning=reasoning)

    return jsonify({
        "content_id": content_id,
        "status": "under_review",
        "message": "Appeal received. This content is now queued for human review.",
        "original_decision": {
            "attribution": content["attribution"],
            "confidence": content["confidence"],
        },
    }), 200


@app.route("/log", methods=["GET"])
def log():
    limit = request.args.get("limit", default=50, type=int)
    return jsonify({"entries": db.get_log(limit)}), 200


@app.route("/analytics", methods=["GET"])
def analytics():
    return jsonify(db.get_analytics()), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "llm_signal": bool(os.environ.get("GROQ_API_KEY"))}), 200


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "rate limit exceeded", "detail": str(e.description)}), 429


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
