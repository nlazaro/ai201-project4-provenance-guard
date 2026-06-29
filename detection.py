"""
detection.py — Provenance Guard detection pipeline.

Three distinct signals, combined into one calibrated confidence score:

  1. llm_signal       (semantic)   — Groq llama-3.3-70b assesses the text holistically.
  2. stylometric_signal (structural)— burstiness, vocabulary diversity, sentence length.
  3. lexical_signal   (lexical)    — density of known AI "tell" phrases / clichés.

Each signal returns a P(AI) in [0, 1] (higher = more AI-like). The combiner produces:
  - ai_likelihood : weighted P(AI)
  - confidence    : how SURE we are of the verdict (not P(AI)); shrinks when signals disagree
  - attribution   : likely_ai | likely_human | uncertain
  - label         : one of three plain-language transparency variants

Design choice — false-positive asymmetry: on a writing platform, calling a human's work
AI is worse than missing some AI. So the "AI" direction needs a wider margin than "human",
and signal disagreement widens the uncertain band rather than forcing a call.
"""
from __future__ import annotations
import os, re, json, math, statistics

# ---- tunable constants (documented in planning.md / README) -------------------
W_LLM, W_STYLO, W_LEXICAL = 0.60, 0.25, 0.15   # ensemble weights (sum = 1.0)
BASE_BAND = 0.12        # half-width of the uncertain band around 0.5 when signals agree
DISAGREE_BAND = 0.28    # extra half-width added in proportion to signal disagreement
AI_MARGIN_BONUS = 0.04  # extra margin required to assert "AI" (false-positive guard)
MIN_WORDS = 25          # below this, text is too short to judge -> force uncertain

# Known LLM stylistic tics / transition clichés (lexical signal).
AI_TELLS = [
    "it is important to note", "it's important to note", "it is worth noting",
    "furthermore", "moreover", "in conclusion", "in summary", "overall",
    "delve", "delving", "tapestry", "navigating", "navigate the", "realm of",
    "paradigm shift", "stakeholders", "leverage", "underscore", "underscores",
    "a testament to", "plays a crucial role", "plays a vital role",
    "in today's world", "in the modern era", "ever-evolving", "ever-changing",
    "it is essential to", "it is equally", "on the other hand", "as a result",
    "transformative", "robust", "seamless", "holistic", "multifaceted",
    "foster", "fostering", "myriad", "pivotal", "comprehensive", "nuanced",
]


# ---- Signal 2: stylometry ----------------------------------------------------
def _sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s.strip()]

def _words(text: str):
    return re.findall(r"[a-zA-Z']+", text.lower())

def stylometric_signal(text: str) -> dict:
    """Structural P(AI): AI text tends to be uniform; human text is bursty/irregular."""
    words = _words(text)
    sents = _sentences(text)
    n_words = len(words)
    if n_words < 5 or not sents:
        return {"score": 0.5, "features": {}, "note": "too short for stylometry"}

    sent_lens = [len(_words(s)) for s in sents] or [n_words]
    mean_len = statistics.mean(sent_lens)
    # Burstiness: coefficient of variation of sentence length. Human ~high, AI ~low.
    cv = (statistics.pstdev(sent_lens) / mean_len) if mean_len else 0.0
    burst_ai = _clamp(1.0 - cv / 0.6)            # low variation -> AI-like

    # Vocabulary diversity (type-token ratio). Very uniform/repetitive OR very polished
    # both deviate from casual human range; we map the "too-clean" high end toward AI.
    ttr = len(set(words)) / n_words
    # Casual human prose clusters ~0.55-0.75 for short samples; >0.8 reads "essayistic".
    ttr_ai = _clamp((ttr - 0.62) / 0.28)         # higher TTR -> a bit more AI-like

    # Contractions / informal lowercase 'i' -> human. Their absence -> more AI-like.
    contractions = len(re.findall(r"\b(i'm|don't|can't|won't|it's|that's|i've|didn't|"
                                  r"wasn't|isn't|gonna|wanna|kinda|ok|lol)\b", text.lower()))
    informal_i = len(re.findall(r"(?<![A-Za-z])i(?![A-Za-z'])", text))  # bare lowercase 'i'
    informal_density = (contractions + informal_i) / max(1, len(sents))
    informal_ai = _clamp(1.0 - informal_density / 1.2)   # lots of informality -> human

    # Average sentence length: long uniform sentences lean AI.
    len_ai = _clamp((mean_len - 10) / 22)

    score = _clamp(0.40 * burst_ai + 0.20 * ttr_ai + 0.25 * informal_ai + 0.15 * len_ai)
    return {
        "score": round(score, 3),
        "features": {
            "sentence_count": len(sents),
            "mean_sentence_len": round(mean_len, 2),
            "burstiness_cv": round(cv, 3),
            "type_token_ratio": round(ttr, 3),
            "informal_density": round(informal_density, 3),
        },
    }


# ---- Signal 3: lexical AI-tells ----------------------------------------------
def lexical_signal(text: str) -> dict:
    low = " " + text.lower() + " "
    hits = [p for p in AI_TELLS if p in low]
    n_words = max(1, len(_words(text)))
    # density per 100 words, squashed to [0,1]
    density = len(hits) / (n_words / 100.0)
    score = _clamp(1 - math.exp(-density / 1.5))
    return {"score": round(score, 3), "hits": hits, "tell_count": len(hits)}


# ---- Signal 1: LLM (Groq) ----------------------------------------------------
def llm_signal(text: str) -> dict:
    """Semantic P(AI) from Groq. Degrades gracefully if the API/key is unavailable."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"score": None, "available": False, "note": "no GROQ_API_KEY; LLM signal skipped"}
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = (
            "You are a forensic writing analyst. Estimate the probability that the TEXT below "
            "was generated primarily by an AI language model (as opposed to written by a human). "
            "Consider coherence patterns, generic phrasing, and stylistic uniformity. "
            'Respond ONLY with JSON: {"ai_probability": <float 0..1>, "reason": "<one short sentence>"}.\n\n'
            f"TEXT:\n{text}"
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=120,
        )
        raw = resp.choices[0].message.content
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        p = float(data.get("ai_probability"))
        return {"score": _clamp(p), "available": True, "reason": data.get("reason", "")}
    except Exception as e:  # network error, bad key, parse failure -> degrade
        return {"score": None, "available": False, "note": f"LLM signal error: {e}"}


# ---- Combination & classification --------------------------------------------
def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def combine(text: str, llm: dict, stylo: dict, lexical: dict) -> dict:
    """Blend available signals into ai_likelihood + confidence + attribution + label."""
    n_words = len(_words(text))

    # Collect (weight, score) for available signals; renormalize if LLM is missing.
    parts = []
    if stylo.get("score") is not None:
        parts.append((W_STYLO, stylo["score"]))
    if lexical.get("score") is not None:
        parts.append((W_LEXICAL, lexical["score"]))
    llm_ok = llm.get("score") is not None
    if llm_ok:
        parts.append((W_LLM, llm["score"]))
    wsum = sum(w for w, _ in parts) or 1.0
    ai_likelihood = sum(w * s for w, s in parts) / wsum

    # Disagreement across available signal scores -> widens the uncertain band.
    scores = [s for _, s in parts]
    disagreement = (max(scores) - min(scores)) if len(scores) > 1 else 0.0

    # Dynamic uncertain band (asymmetric: AI side is wider = harder to assert).
    half = BASE_BAND + DISAGREE_BAND * disagreement
    hi_thresh = 0.5 + half + AI_MARGIN_BONUS      # must clear this to be "likely_ai"
    lo_thresh = 0.5 - half                         # must fall below this to be "likely_human"

    too_short = n_words < MIN_WORDS
    if too_short:
        attribution = "uncertain"
    elif ai_likelihood >= hi_thresh:
        attribution = "likely_ai"
    elif ai_likelihood <= lo_thresh:
        attribution = "likely_human"
    else:
        attribution = "uncertain"

    # Confidence = certainty in the verdict, reduced by disagreement and short length.
    raw_conf = abs(ai_likelihood - 0.5) * 2
    confidence = _clamp(raw_conf * (1 - 0.5 * disagreement))
    if too_short:
        confidence = min(confidence, 0.30)
    if not llm_ok:
        confidence = min(confidence, 0.60)        # weaker without the semantic signal
    confidence = round(confidence, 3)

    label = make_label(attribution, confidence, ai_likelihood)
    return {
        "attribution": attribution,
        "confidence": confidence,
        "ai_likelihood": round(ai_likelihood, 3),
        "disagreement": round(disagreement, 3),
        "signals": {
            "llm_score": llm.get("score"),
            "stylometric_score": stylo.get("score"),
            "lexical_score": lexical.get("score"),
            "llm_available": llm_ok,
        },
        "label": label,
    }


# ---- Transparency labels (three variants) ------------------------------------
LABELS = {
    "high_ai": (
        "🤖 Likely AI-generated. Our automated check estimates roughly a {pct}% likelihood that "
        "this text was produced with substantial help from an AI writing tool. This is an "
        "automated estimate, not a certainty — if you wrote this yourself, you can appeal and a "
        "human will review it."
    ),
    "high_human": (
        "✍️ Likely human-written. Our automated check estimates roughly a {pct}% likelihood that "
        "a person wrote this text; no strong AI-generation signals stood out. Automated estimates "
        "can still be wrong, and any creator may appeal a result."
    ),
    "uncertain": (
        "❓ Uncertain. Our automated check could not reach a confident conclusion about whether this "
        "text was written by a person or an AI tool — the signals were mixed or too weak to call. "
        "We are showing this honestly rather than guessing; no attribution is being asserted."
    ),
}

def make_label(attribution: str, confidence: float, ai_likelihood: float) -> dict:
    """Pick the variant by attribution; AI/human variants display the estimated likelihood."""
    if attribution == "likely_ai":
        variant, pct = "high_ai", int(round(ai_likelihood * 100))
    elif attribution == "likely_human":
        variant, pct = "high_human", int(round((1 - ai_likelihood) * 100))
    else:
        variant, pct = "uncertain", None
    text = LABELS[variant].format(pct=pct) if pct is not None else LABELS[variant]
    return {"variant": variant, "text": text}


def analyze(text: str) -> dict:
    """Full pipeline for a piece of text (no I/O)."""
    llm = llm_signal(text)
    stylo = stylometric_signal(text)
    lexical = lexical_signal(text)
    result = combine(text, llm, stylo, lexical)
    result["_debug"] = {"llm": llm, "stylo": stylo, "lexical": lexical}
    return result
