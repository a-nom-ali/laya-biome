#!/usr/bin/env python3
"""
label_with_nimble.py — C# raw_states.json -> Nimble (local, Ollama 0.35+ qwen3.5-9B) -> laya_terrain.jsonl
- Keeps choice ≤12 (head budget 256/12=21 tokens/option) via per-biome filtering
- Reserves Jev $1 for golden validation (validate_with_jev.py)
- Seeded-deterministic: laya's probabilities are sampled per tile, then cached

Usage:
  python scripts/label_with_nimble.py --input data/raw_states.json --output data/laya_terrain.jsonl --nimble http://localhost:11434 --model nimble
  # Generates raw_states.json via C# HeightmapGen -> export JSON (see csharp/README)
"""
import argparse, json, pathlib, random, time, sys
import requests

# Per-biome flora vocab (keeps choice ≤12, filtered by macro terrain_type)
FLORA_BY_TERRAIN = {
    "Plains": ["Grassland", "Savanna Grass", "Shrub Steppe", "Farmland"],
    "Hills": ["Deciduous Scrub", "Pine Scrub", "Grassland Hills"],
    "Mountains": ["Pine Variants", "Alpine Meadow", "Lichen", "Bare Rock", "Scrub Brush"],
    "Canyons": ["Desert Scrub", "Riparian Cottonwood", "Bare Sandstone"],
    "Dunes": ["Sparse Dune Grass", "Bare Sand", "Oasis Palm"],
}

def nimble_predict(state, questions, endpoint, model):
    # Nimble shares Jev's /v1/systemone shape (engine/laya_gateway.py:86)
    payload = {"state": state, "questions": questions}
    # Ollama Nimble exposes /api/generate with SystemOne, or /v1/systemone if proxied via laya_gateway
    # Try /v1/systemone first, fallback to /api/generate
    for url in [f"{endpoint}/v1/systemone", f"{endpoint}/api/generate"]:
        try:
            r = requests.post(url, json={**payload, "model": model}, timeout=60)
            if r.status_code == 200:
                j = r.json()
                # Normalize to {answers: {...}}
                if "answers" in j:
                    return j
                # Ollama's /api/generate returns {"response": ...}
                if "response" in j:
                    return json.loads(j["response"])
        except Exception as e:
            continue
    raise RuntimeError(f"Nimble failed at {endpoint} for state {state}")

def build_questions(macro_terrain=None):
    # Schema A: macro params (blended)
    q_macro = {
        "moisture": {"type": "score", "instructions": "Moisture level 0 Arid to 5 Saturated, given rainfall+distance_to_ocean+windward", "criteria": ["Arid","Dry","Semi-arid","Moderate","Humid","Saturated"]},
        "temperature_band": {"type": "score", "instructions": "Temperature band 0 Freezing to 5 Scorching, given latitude+altitude+ocean_current", "criteria": ["Freezing","Cold","Cool","Mild","Warm","Scorching"]},
        "base_terrain": {"type": "choice", "instructions": "Base terrain, choose one", "criteria": {k: k for k in ["Plains","Hills","Mountains","Canyons","Dunes"]}},
        "coastal_proximity": {"type": "noul", "instructions": "Is distance_to_ocean <0.05?", "criteria": {"true": "coastal", "false": "inland"}},
    }
    # Schema B: micro (filtered)
    flora_opts = FLORA_BY_TERRAIN.get(macro_terrain, ["Grassland","Scrub","Bare"])
    q_micro = {
        "canopy_density": {"type": "score", "instructions": "Canopy 0 Barren to 3 Dense, given moisture+terrain", "criteria": ["Barren","Sparse","Moderate","Dense"]},
        "primary_flora": {"type": "choice", "instructions": f"Primary flora for {macro_terrain} (≤12)", "criteria": {k: k for k in flora_opts}},
        "apex_predator": {"type": "noul", "instructions": "Probability dangerous creature spawns here?"},
    }
    return q_macro, q_micro

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw_states.json")
    ap.add_argument("--output", default="data/laya_terrain.jsonl")
    ap.add_argument("--nimble", default="http://localhost:11434")
    ap.add_argument("--model", default="nimble")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    raw = json.loads(pathlib.Path(args.input).read_text())
    if args.limit:
        raw = raw[:args.limit]
    # Balance check before labeling
    print(f"[label] {len(raw)} raw states from {args.input}")

    out = []
    for idx, state in enumerate(raw):
        # Macro
        q_macro, _ = build_questions()
        try:
            res_macro = nimble_predict(state, q_macro, args.nimble, args.model)
        except Exception as e:
            print(f"[warn] macro {idx} failed: {e}", file=sys.stderr)
            continue
        # Extract macro terrain for micro filtering
        macro_terrain = res_macro.get("answers", res_macro).get("base_terrain", {}).get("choice", "Plains")
        # Micro (conditioned on macro)
        _, q_micro = build_questions(macro_terrain)
        micro_state = {**state, "macro_moisture": res_macro.get("answers", res_macro).get("moisture", {}).get("score", 2), "macro_terrain": macro_terrain}
        try:
            res_micro = nimble_predict(micro_state, q_micro, args.nimble, args.model)
        except Exception as e:
            print(f"[warn] micro {idx} failed: {e}", file=sys.stderr)
            continue

        # Merge into laya training entry (two tasks, one model)
        for task, qs, res in [("macro_params", q_macro, res_macro), ("micro_flora", q_micro, res_micro)]:
            entry = {
                "task": task,
                "state": micro_state if task=="micro_flora" else state,
                "questions": qs,
                "gold": res.get("answers", res)  # Nimble's probabilities are gold for distillation
            }
            out.append(entry)
        if idx % 100 == 0:
            print(f"[label] {idx}/{len(raw)}")

    # Balance validation (as pasted Step 3)
    from collections import Counter
    cnt = Counter()
    for e in out:
        for qid, ans in e["gold"].items():
            if ans.get("choice"):
                cnt[ans["choice"]] += 1
    print(f"[balance] {cnt}")
    # Warn if any label <5% of max
    if cnt:
        mx = max(cnt.values())
        for k, v in cnt.items():
            if v < mx * 0.05:
                print(f"[warn] {k} under-represented {v}/{mx} — duplicate or generate more", file=sys.stderr)

    pathlib.Path(args.output).write_text("\n".join(json.dumps(x) for x in out))
    print(f"[done] wrote {len(out)} entries (2 tasks per raw) to {args.output}")

if __name__ == "__main__":
    main()
