#!/usr/bin/env python3
"""
validate_with_jev.py — Golden 200 vs Jev ($1) to check Nimble calibration
- Samples 200 states from laya_terrain.jsonl, runs both Nimble and Jev, reports agreement
- Use TYPESAFE_API_KEY with $1 credit; stops at $1

Usage:
  TYPESAFE_API_KEY=... python scripts/validate_with_jev.py --input data/laya_terrain.jsonl --sample 200
"""
import argparse, json, pathlib, os, sys
import requests

def jev_predict(state, questions):
    try:
        from typesafe_sdk import TypeSafeClient
    except ImportError:
        print("typesafe-sdk not installed: pip install typesafe-sdk", file=sys.stderr)
        sys.exit(1)
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        print("TYPESAFE_API_KEY required", file=sys.stderr)
        sys.exit(1)
    client = TypeSafeClient(api_key=api_key, model="jev-latest", timeout=30)
    return client.system_one(state=state, questions=questions)

def nimble_predict(state, questions, endpoint="http://localhost:11434", model="nimble"):
    payload = {"state": state, "questions": questions}
    for url in [f"{endpoint}/v1/systemone", f"{endpoint}/api/generate"]:
        try:
            r = requests.post(url, json={**payload, "model": model}, timeout=60)
            if r.status_code == 200:
                j = r.json()
                if "answers" in j:
                    return j
                if "response" in j:
                    return json.loads(j["response"])
        except Exception:
            continue
    raise RuntimeError("Nimble failed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/laya_terrain.jsonl")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--nimble", default="http://localhost:11434")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    sample = rows[:args.sample]
    print(f"[validate] {len(sample)} / {len(rows)}")

    agree_choice = []
    mae_score = []
    cost_est = 0
    for r in sample:
        state = r["state"]
        questions = r["questions"]
        # Nimble
        try:
            n_res = nimble_predict(state, questions, args.nimble)
            n_ans = n_res.get("answers", n_res)
        except Exception as e:
            print(f"[warn] nimble failed {e}")
            continue
        # Jev (counts against $1 — ~200 * ~1k tokens = 200k = $0.008)
        try:
            j_res = jev_predict(state, questions)
            j_ans = j_res.answers if hasattr(j_res, "answers") else j_res.get("answers", {})
        except Exception as e:
            print(f"[warn] jev failed {e}")
            continue

        for qid, qdef in questions.items():
            if qdef["type"] == "choice":
                if n_ans.get(qid, {}).get("choice") == j_ans.get(qid, {}).get("choice"):
                    agree_choice.append(1)
                else:
                    agree_choice.append(0)
            elif qdef["type"] == "score":
                n_s = n_ans.get(qid, {}).get("score", 0)
                j_s = j_ans.get(qid, {}).get("score", 0) if isinstance(j_ans.get(qid), dict) else j_ans.get(qid, {}).get("score", 0)
                try:
                    mae_score.append(abs(float(n_s) - float(j_s)))
                except Exception:
                    pass

    if agree_choice:
        print(f"[result] choice agreement {sum(agree_choice)/len(agree_choice):.3f} ({sum(agree_choice)}/{len(agree_choice)}) — want >0.85")
    if mae_score:
        import numpy as np
        print(f"[result] score MAE {np.mean(mae_score):.3f} — want <0.6")
    print(f"[result] estimated Jev cost ~${len(sample)*0.00004:.4f} (at $0.042/M input)")

if __name__ == "__main__":
    main()
