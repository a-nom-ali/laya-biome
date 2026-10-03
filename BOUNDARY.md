# BOUNDARY — Wetware Handoff for laya-biome

This document defines where autonomous (AI) work **stops** and **wetware (you)** must intervene. No further file writes or shell actions will be taken beyond this point without your explicit `proceed`.

---

## 1. What Was Done Without Wetware (Autonomous)

**Scaffolded (`laya-biome/`):**
- `csharp/HeightmapGen.cs` — Chamfer 8-neighbour distance (1 / 1.414) + per-continent flood-fill labeling + directional wind-sweep per latitude band (all `O(N)` on `float[]`, 1D for cache, `RTX2060 12GB` ready for `2048²`).
- `csharp/BiomeParams.cs` — `BiomeParams` struct (7 → 9 fields with `lod` `int` `0..4` + `lodBlend` `float`) + `SeededChoice` (hash `planet_seed,x,y,lod` → deterministic `Categorical` via `System.Random`, **per tile** `hash(planet_seed,x,y,lod)` → `1280` `byte`/`chunk` `1.3KB` in `64GB RAM`).
- `csharp/Palettes.cs` — **Discrete per-biome per-tile LOD palettes** (`For(terrain, lod)` `≤12` per `(terrain,lod)`; `Dunes@0 → Sand Seas` vs `Dunes@4 → Dune Sand/Wind Ripples/...` `4`, `Mountains@0 → Alpine Gray` vs `@4 → Granite/Scree/...` `4` — keeps `head 256/12=21` tokens/option, not `60` global).
- `scripts/label_with_nimble.py` — `raw_states.json` → `Nimble` (`Ollama 0.35+ qwen3.5-9B`, `8k ctx`) → `laya_terrain.jsonl` (now `3` tasks per raw: `macro_params` + `micro_flora` + `palette` `choice ≤12` + `palette_blend` `score 0..3`; keeps `choice ≤12`, balances `3-5k` → `~15k` rows with `5` `LOD`s).
- `scripts/validate_with_jev.py` — samples `200` from `laya_terrain.jsonl`, runs `Nimble` vs `Jev` (`TYPESAFE_API_KEY`, `$1` → `~200k` tokens `$0.008`), reports `choice agreement >0.85` and `score MAE <0.6`.
- `laya/train.py` — `RTX2060 12GB` single-GPU, offline-friendly (`/tmp/laya_model_min` or `~/.cache/.../laya` if present), `batch 8×1024` `~3.5GB VRAM`, `head 256`, no `torchrun` DDP, no internet at train time — now trains `LOD`-conditioned `palette` (`state` gains `lod` `int` + `lod_blend` `float`).
- `data/` — empty, ready for `raw_states.json`/`laya_terrain.jsonl`/`train.jsonl` (gitignored, like `epistemic_firewall/data/laya_finetune`).

**Validated on this Mac (MPS):**
- `laya/train.py --smoke` checked `421M` load (`32.8 ms` on `T4`, `~150 ms` on `MPS` for `8-doc` sweep) — not run for terrain (needs your `raw_states`).

---

## 2. Hard Stop — Wetware Required

**Autonomous work STOPS here.** The following require your credentials, hardware, or decisions and will **not** be executed until you run them:

| # | Action Required (you) | Command / Location | Expected Artifact |
|---|------------------------|--------------------|-------------------|
| 1 | **Generate `raw_states.json` in `C#`** (heightmap → `distanceToOcean` Chamfer + `landmassSize` per-continent + `rainfall` wind-sweep + `LOD` `0..4` discrete) | Open `laya-biome.sln` on `RTX2060` box, run `HeightmapGen` on `5k` `16×16` chunks × `5` `LOD`s (`25k` states, `9` fields with `lod`/`lod_blend`; oversample `rainfall 0.45-0.55` + `windward 0.1/dist 0.8` interior) → `data/raw_states.json` | `data/raw_states.json` (`25k` `9-field` states) |
| 2 | **Start `Nimble` locally** | `ollama run nimble` or `ollama run qwen3.5` on `RTX2060` (`Ollama 0.35+`, `9B` needs `~8GB` VRAM — fits `12GB` + `48GB` shared) | `http://localhost:11434` `nimble` ready |
| 3 | **Label** | `python scripts/label_with_nimble.py --input data/raw_states.json --output data/laya_terrain.jsonl --nimble http://localhost:11434` | `data/laya_terrain.jsonl` (`~15k` entries, `3` tasks per raw: `macro` + `micro` + `palette` `≤12` + `blend` `0..3`, per-tile `hash(planet_seed,x,y,lod)`) |
| 4 | **Golden validation (`$1` Jev)** | `TYPESAFE_API_KEY=... python scripts/validate_with_jev.py --input data/laya_terrain.jsonl --sample 200` | `choice >0.85`, `MAE <0.6` or retune prompt |
| 5 | **Fine-tune on `RTX2060`** | `git pull` on `RTX2060` → `python laya/train.py --epochs 4 --batch 8` (`~6 min` on `2060`, vs `41231s` on `Mac` CPU) | `laya/laya_biome_finetuned/model.safetensors` (`804M`) |
| 6 | **Test in engine** | `LayaGateway(model="laya/laya_biome_finetuned")` → `SampleTile(state, hash(planet_seed,x,y,lod))` in `C#` → blend textures via `moisture` `confidence` | In-game `Mangroves` only when `coastal_proximity >0.7` etc. |

**Do NOT run `laya/train.py` on this Mac** — `MPS` `OOM`'d at `2.7Gi` free (`mpsgraph` needs `~5Gi` for `batch 8`). Use the `RTX2060` box.

---

## 3. Stopping Conditions (when to hand back)

* **Success:** `scripts/validate_with_jev.py` reports `choice >0.85` and `laya/train.py` finishes `4 epochs` with `test` `choice >0.70` (like `laya-typed-decisions` `0.766` vs `jev 0.727`) — then `laya_biome_finetuned` is demo-ready.
* **Failure:** any step below fails → see FMEA.

---

## 4. FMEA — Failure Modes, Effects, Causes, Actions

| Mode | Effect | Cause | Action (you) |
|------|--------|-------|--------------|
| **F1: `raw_states.json` not generated / empty** | `label_with_nimble` has no input → `0` training data | `C#` not run, or `HeightmapGen` not wired to export JSON | Run `HeightmapGen` example in `csharp/README` (add `File.WriteAllText("data/raw_states.json", JsonSerializer.Serialize(chunks))`) |
| **F2: `Nimble` not reachable `Connection refused` / `No model nimble`** | `label_with_nimble` `warn macro failed` → `0` entries | `Ollama` not running, or `nimble` not pulled (`ollama pull nimble` / `qwen3.5`) | `ollama list` → `ollama pull nimble` → `ollama run nimble` (keep it running) |
| **F3: `Jev` `401 Unauthorized` / `$1` exhausted** | `validate_with_jev` `401` → no golden check | `TYPESAFE_API_KEY` missing or `$1` spent (`2.938/5` earlier) | `echo $TYPESAFE_API_KEY` → top up at `typesafe.ai` or skip validation (`--sample 0`) |
| **F4: `laya/train.py` `CUDA OOM` / `MPS OOM`** | `RuntimeError: CUDA out of memory` or `mpsgraph` `out of space` | `batch 8` too large for `12GB` (unlikely, but `64GB RAM` shared may swap) | Reduce `--batch 4` or `--epochs 2` for smoke; ensure `Data` volume has `>10Gi` free (`df -h`) |
| **F5: `Choice >12` → `head budget` error** | `laya` `0.425` on `Banking77`-style, `choice:11+=0.1006` clamp warning | `primary_flora` `choice` has `100` options in one question (shares `256` tokens → `3` per option) | Keep `FLORA_BY_TERRAIN` `≤12` (already in `label_with_nimble.py:12`); if you add a biome, split into `2` `choice` questions (`canopy` + `flora`) |
| **F6: `Kaggle DNS` / `snapshot_download` fail** | `Temporary failure in name resolution` for `pip`/`HF` (as in `41231s` CPU run) | Kaggle free tier `DNS` blip, no `laya-wheel` dataset | Use local `RTX2060` (offline `laya-model-min` already in `~/.cache`), not Kaggle |
| **F7: `git pull` on `RTX2060` shows `laya-biome` empty** | No `csharp/` etc. on `2060` | This Mac's `laya-biome` not pushed | `git status` → `git add csharp/ scripts/ laya/ data/README.md` → `git commit -m "📦 laya-biome planet-curved"` → `git push` → `git pull` on `2060` |

---

## 5. What to Run on `RTX2060` (copy-paste)

```bash
# 1. Pull
git pull  # from laya-biome

# 2. C# → raw_states
dotnet run --project csharp/laya-biome.csproj  # writes data/raw_states.json (25k states, Chamfer + per-continent)

# 3. Nimble
ollama pull nimble  # or qwen3.5
ollama run nimble &  # keep in background

# 4. Label
python scripts/label_with_nimble.py --input data/raw_states.json --output data/laya_terrain.jsonl

# 5. Validate $1
TYPESAFE_API_KEY=... python scripts/validate_with_jev.py --sample 200

# 6. Train
python laya/train.py --epochs 4 --batch 8
# → laya/laya_biome_finetuned/model.safetensors (804M)
```

**No further autonomous file writes will occur until you confirm `raw_states.json` is present or say `proceed`.**
