#!/usr/bin/env python3
"""
laya-biome train — RTX2060 12GB (64GB RAM) single-GPU, offline
- No Kaggle, no torchrun DDP, no internet at train time (uses local laya-model-min if present)
- Single model, mixed tasks: macro_params + micro_flora in same train.jsonl

Usage on RTX2060 box:
  git pull  # from laya-biome
  pip install laya transformers datasets safetensors huggingface_hub accelerate
  python laya/train.py --epochs 4 --batch 8
  # Or smoke:
  python laya/train.py --smoke

Hardware: RTX2060 12GB + 48GB shared + 64GB RAM → batch 8×1024 fits in ~3.5GB VRAM.
This Mac (MPS) can run --smoke only (needs ~5GB free on Data volume).
"""
import argparse, json, pathlib, time, random, os, sys
import torch

# RTX2060: prefer CUDA, fallback to CPU/MPS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def smoke():
    import pathlib as p
    for f in ["data/raw_states.json", "data/laya_terrain.jsonl", "data/train.jsonl"]:
        pp = p.Path(f)
        print(f"{'OK' if pp.exists() else 'MISSING'} {f} {pp.stat().st_size if pp.exists() else 0}")
    try:
        import laya
        print(f"laya {laya.__version__} device {DEVICE}")
    except Exception as e:
        print(f"laya import failed: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="check data + laya import, no train")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--data", default="data/laya_terrain.jsonl")
    ap.add_argument("--output", default="laya/laya_biome_finetuned")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    # --- Check data ---
    data_path = pathlib.Path(args.data)
    if not data_path.exists():
        print(f"Missing {data_path} — run: python scripts/label_with_nimble.py --input data/raw_states.json --output {args.data}", file=sys.stderr)
        sys.exit(1)

    # --- Load laya model (offline if present) ---
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_sequence, render_options, QTYPES, build_model, proper_reward
    from safetensors.torch import load_file, save_file

    # Prefer local model (pushed to laya-model-min-1-2g or HF cache)
    model_dir = None
    for cand in [pathlib.Path("/tmp/laya_model_min"), pathlib.Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982"]:
        if cand.exists():
            model_dir = str(cand)
            print(f"Using local model at {model_dir}")
            break
    if model_dir is None:
        print("Fetching convaiinnovations/laya from HF...")
        model_dir = snapshot_download("convaiinnovations/laya")
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256  # 256/12=21 tokens/option, safe for ≤12
    print(f"cfg max_len {cfg['max_len']} head {cfg['head_max_len']} device {DEVICE}")

    # --- Build training items (same as Kaggle) ---
    rows = [json.loads(l) for l in open(data_path) if l.strip()]
    print(f"rows {len(rows)} (each has 2 tasks → {len(rows)*2} decisions)")

    def build_item(state, q, gold_q):
        t = q["type"]
        crit = q.get("criteria", {})
        if t == "choice":
            keys = list(crit.keys())
            target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
        elif t == "noul":
            target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
        else:  # score
            n = len(crit) if isinstance(crit, list) else 6
            target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n)]
        s = sum(target)
        target = [v/s for v in target] if s>0 else [1/len(target)]*len(target)
        label = target.index(max(target))
        k = len(render_options({"t": t, "crit": crit}))
        seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit}, cfg["max_len"], cfg["head_max_len"])
        if len(markers) != k:
            return None
        return {"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target, "label": label}

    items = []
    for r in rows:
        state = r["state"]
        # r has "questions" and "gold" per the label_with_nimble schema (task is inside, but we flattened)
        # Our label_with_nimble writes {"task":..., "state":..., "questions":..., "gold":...} per task
        # So each row is already one task; handle both shapes
        if "questions" in r and "gold" in r:
            qs = r["questions"]
            gold = r["gold"]
            for qid, q in qs.items():
                if qid in gold:
                    it = build_item(state, q, gold[qid])
                    if it:
                        items.append(it)
        else:
            # Fallback: row is state only
            pass
    print(f"items {len(items)}")

    # --- Model ---
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.to(DEVICE)
    model.train()

    # RTX2060 12GB: batch 8, accum 2 => effective 16
    EPOCHS = args.epochs
    MICRO_BATCH = args.batch
    GRAD_ACCUM = 2
    GROUP_SIZE = 4
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, (len(items)//(MICRO_BATCH*GRAD_ACCUM))*EPOCHS))

    def collate(chunk, pad_id):
        n, L = len(chunk), max(len(it["ids"]) for it in chunk)
        kmax = max(len(it["markers"]) for it in chunk)
        ids = torch.full((n, L), pad_id, dtype=torch.long)
        att = torch.zeros((n, L), dtype=torch.long)
        mpos = torch.zeros((n, kmax), dtype=torch.long)
        mmask = torch.zeros((n, kmax), dtype=torch.bool)
        target = torch.zeros((n, kmax), dtype=torch.float32)
        for i, it in enumerate(chunk):
            ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
            att[i, :len(it["ids"])] = 1
            k = len(it["markers"])
            mpos[i, :k] = torch.tensor(it["markers"])
            mmask[i, :k] = True
            target[i, :len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
        return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target, "qtype": torch.tensor([it["qtype"] for it in chunk])}

    t0 = time.time()
    for epoch in range(EPOCHS):
        random.shuffle(items)
        epoch_loss = 0
        n_batches = 0
        for b_idx in range(0, len(items), MICRO_BATCH):
            chunk = items[b_idx:b_idx+MICRO_BATCH]
            batch = collate(chunk, tok.pad_token_id)
            # Autocast only on CUDA
            ctx = torch.autocast("cuda", dtype=torch.float16) if DEVICE.type == "cuda" else torch.amp.autocast("cpu")
            with ctx:
                logits, _ = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["marker_pos"].to(DEVICE), batch["marker_mask"].to(DEVICE), batch["qtype"].to(DEVICE))
            # Simplified loss: CE only for RTX2060 smoke (RLCD needs GROUP_SIZE sampling; keep for full)
            loss = -(batch["target"].to(DEVICE) * torch.log_softmax(logits, -1)).sum(-1).mean() / GRAD_ACCUM
            loss.backward()
            if (n_batches+1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            if n_batches % 20 == 0:
                print(f" epoch {epoch+1}/{EPOCHS} step {n_batches} loss {loss.item()*GRAD_ACCUM:.4f}")
        print(f"=== Epoch {epoch+1}/{EPOCHS} done {time.time()-t0:.1f}s avg {epoch_loss/max(1,n_batches):.4f} ===")

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.cpu() for k,v in model.state_dict().items()}, str(out / "model.safetensors"))
    print(f"Saved to {out} — load via laya.load('{out}') or LayaGateway(model='{out}')")

if __name__ == "__main__":
    main()
