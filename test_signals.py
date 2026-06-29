"""
test_signals.py — Milestone 4 verification harness.

Run this AFTER setting GROQ_API_KEY in your .env. It runs the full three-signal
pipeline on the milestone's four canonical inputs and prints every signal plus the
combined result, so you can confirm scores vary meaningfully and all three label
variants are reachable. Copy two rows into the README confidence-scoring section.

    python test_signals.py
"""
from dotenv import load_dotenv
load_dotenv()
import detection as d

INPUTS = {
 "clearly_AI": "Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications. Furthermore, stakeholders across various sectors must collaborate to ensure responsible deployment.",
 "clearly_human": "ok so i finally tried that new ramen place downtown and honestly? underwhelming. the broth was fine but they put WAY too much sodium in it and i was thirsty for like three hours after. my friend got the spicy version and said it was better. probably won't go back unless someone drags me there",
 "formal_human_econ": "The relationship between monetary policy and asset price inflation has been extensively studied in the literature. Central banks face a fundamental tension between their mandate for price stability and the unintended consequences of prolonged low interest rates on equity and real estate valuations.",
 "edited_AI": "I've been thinking a lot about remote work lately. There are genuine tradeoffs — flexibility and no commute on one side, isolation and blurred work-life boundaries on the other. Studies show productivity varies widely by individual and role type.",
}

hdr = f"{'input':18s} {'llm':>5} {'stylo':>6} {'lex':>5} | {'aiLike':>6} {'conf':>6} {'disagr':>7}  {'attribution':13s} variant"
print(hdr); print("-"*len(hdr))
for name, txt in INPUTS.items():
    r = d.analyze(txt)
    s = r["signals"]
    llm = s["llm_score"] if s["llm_score"] is not None else float("nan")
    print(f"{name:18s} {llm:5.2f} {s['stylometric_score']:6.2f} {s['lexical_score']:5.2f} | "
          f"{r['ai_likelihood']:6.2f} {r['confidence']:6.2f} {r['disagreement']:7.2f}  "
          f"{r['attribution']:13s} {r['label']['variant']}")
    if not s["llm_available"]:
        print("    (LLM unavailable — set GROQ_API_KEY to enable the semantic signal)")
