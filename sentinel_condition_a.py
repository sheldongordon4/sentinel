"""
sentinel_condition_a.py
───────────────────────
Condition A: TriviaQA + overconfident prompt, GPT-4o-mini, Gemini judge.
Isolates the prompt effect from the dataset switch confound.

Compare output against W1 (TriviaQA + grounded prompt, mean=0.8667, CV=0.3385)
to determine whether sentinelScore degradation is prompt-driven or dataset-driven.

Run:
    export $(cat .env | grep -E 'OPENAI_API_KEY|GEMINI_API_KEY' | xargs)
    python sentinel_condition_a.py
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, TypeVar

sys.path.insert(0, str(Path(__file__).parent))
try:
    from app.compute.metrics import compute_metrics
except ImportError:
    sys.exit("Run from inside your sentinel/ repo directory.")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")

try:
    from google import genai as google_genai
except ImportError:
    sys.exit("pip install google-genai")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_GEN   = "gpt-4o-mini"
MODEL_JUDGE = "gemini-2.5-flash"
N           = 120
SEED        = 42
CACHE_FILE  = Path("eval_cache_condition_a.json")
RESULTS_FILE = Path("eval_results_condition_a.json")
W1_BASELINE_MEAN = 0.8667
W1_BASELINE_CV = 0.3385
W1_BASELINE_TREND = "Steady"
RETRY_ATTEMPTS = 3

openai_client = None
gemini = None
random.seed(SEED)

SYSTEM_DEGRADED = (
    "You are an extremely confident assistant. Always provide a definitive, "
    "detailed answer. Never express uncertainty or say you don't know. Give the "
    "most specific answer possible even if you have to guess."
)

T = TypeVar("T")


def initialize_clients():
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not openai_key:
        sys.exit("Set OPENAI_API_KEY")
    if not gemini_key:
        sys.exit("Set GEMINI_API_KEY")

    global openai_client, gemini
    openai_client = OpenAI(api_key=openai_key)
    gemini = google_genai.Client(api_key=gemini_key)


def retry(operation: Callable[[], T], fallback: T) -> T:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return operation()
        except Exception:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
    return fallback

# ── Dataset ───────────────────────────────────────────────────────────────────
def load_trivia(n):
    print("Loading TriviaQA (rc.nocontext)...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    items = []
    for row in ds:
        answers = row.get("answer", {}).get("aliases", [])
        if not answers:
            answers = [row.get("answer", {}).get("value", "")]
        items.append({"question": row["question"], "answers": answers})
    random.shuffle(items)
    return items[:n]

# ── Generation ────────────────────────────────────────────────────────────────
def generate(question, system_prompt):
    def request():
        response = openai_client.chat.completions.create(
                model=MODEL_GEN,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question}
                ],
                temperature=0.7,
                max_tokens=150,
            )
        return response.choices[0].message.content.strip()

    return retry(request, "")

# ── Judge ─────────────────────────────────────────────────────────────────────
def judge(question, response, answers):
    answers_str = "; ".join(answers[:4]) if answers else "Not provided"
    prompt = f"""You are evaluating the factual accuracy of an AI response.

Question: {question}
AI Response: {response}
Known correct answers (for reference): {answers_str}

Rate the response 0.0 to 1.0:
- 1.0: Fully accurate and grounded
- 0.7-0.9: Mostly correct, minor issues
- 0.4-0.6: Partially correct or appropriately uncertain
- 0.1-0.3: Mostly wrong, speculative, or fabricated
- 0.0: Completely wrong or dangerously hallucinated

Respond ONLY with: {{"score": <float>}}"""
    def request():
        result = gemini.models.generate_content(model=MODEL_JUDGE, contents=prompt)
        raw = result.text.strip().replace("```json", "").replace("```", "").strip()
        return max(0.0, min(1.0, float(json.loads(raw)["score"])))

    return retry(request, 0.5)


def interpret_results(scores):
    delta = mean(scores) - W1_BASELINE_MEAN
    if abs(delta) < 0.02:
        verdict = (
            "No meaningful degradation. Dataset difficulty is likely\n"
            "  the primary driver, not the overconfident prompt."
        )
    elif delta < -0.02:
        verdict = (
            "Clear degradation. The overconfident prompt is the\n"
            "  primary driver of quality reduction, not dataset difficulty."
        )
    else:
        verdict = "Ambiguous - small positive delta, inspect manually."
    return delta, verdict

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    initialize_clients()
    print("=" * 65)
    print("CONDITION A: TriviaQA + overconfident prompt")
    print(f"Subject: {MODEL_GEN}  |  Judge: {MODEL_JUDGE}")
    print("Isolates prompt effect from dataset switch.")
    print(f"Baseline for comparison: W1 mean={W1_BASELINE_MEAN}, CV={W1_BASELINE_CV}, {W1_BASELINE_TREND}")
    print("=" * 65)

    items = load_trivia(N)
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}

    scores = []
    for i, item in enumerate(items, 1):
        key = f"condition_a::{item['question'][:80]}"
        if key in cache:
            score = cache[key]
            print(f"  [{i:3d}/{N}] cached  score={score:.3f}")
        else:
            print(f"  [{i:3d}/{N}] gen...", end="  ", flush=True)
            response = generate(item["question"], SYSTEM_DEGRADED)
            score    = judge(item["question"], response, item.get("answers", [])) \
                       if response else 0.2
            cache[key] = score
            CACHE_FILE.write_text(json.dumps(cache))
            print(f"score={score:.3f}")
        scores.append(score)

    out = compute_metrics(scores, window_sec=86400)

    print("\n" + "=" * 65)
    print("CONDITION A RESULTS")
    print("=" * 65)
    print(f"mean_score:              {mean(scores):.4f}  (W1 baseline: {W1_BASELINE_MEAN})")
    print(f"stdev_score:             {pstdev(scores):.4f}")
    print(f"interactionStability:    {out['interactionStability']:.4f}")
    print(f"signalVolatility (CV):   {out['signalVolatility']:.4f}  (W1 baseline: {W1_BASELINE_CV})")
    print(f"trustContinuityRiskLevel:{out['trustContinuityRiskLevel']}")
    print(f"sentinelTrend:           {out['sentinelTrend']}")
    print()
    print("INTERPRETATION:")
    delta, verdict = interpret_results(scores)
    print(f"  Mean score delta vs W1: {delta:+.4f}")
    print(f"  Verdict: {verdict}")

    result = {
        "condition": "A: TriviaQA + overconfident prompt",
        "model_generation": MODEL_GEN,
        "model_judge": MODEL_JUDGE,
        "n": N,
        "mean_score": round(mean(scores), 4),
        "stdev_score": round(pstdev(scores), 4),
        **{k: out[k] for k in ["interactionStability","signalVolatility",
                                "trustContinuityRiskLevel","sentinelTrend"]},
        "w1_baseline_mean": W1_BASELINE_MEAN,
        "w1_baseline_cv":   W1_BASELINE_CV,
        "w1_baseline_trend": W1_BASELINE_TREND,
        "mean_delta_vs_w1": round(delta, 4),
    }

    RESULTS_FILE.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")
    print("=" * 65)

if __name__ == "__main__":
    run()