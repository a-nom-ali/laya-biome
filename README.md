# laya-biome — Planet-curved Laya for Biome / Flora / Fauna

**Single `laya` (`421M`) checkpoint, hierarchical `state → questions` for a planet demo.** See `BOUNDARY.md` for the hard stop before wetware (you) runs `C#`, `Nimble`, and `RTX2060` training.

* `csharp/HeightmapGen.cs` — Chamfer `1/1.414` distance + per-continent `landmassSize` + wind-sweep `rainfall` (`O(N)` on `float[]`, `12GB VRAM` ready)
* `scripts/label_with_nimble.py` — `raw_states.json` → `Nimble` (`Ollama 0.35+ qwen3.5-9B`) → `laya_terrain.jsonl` (`choice ≤12`)
* `scripts/validate_with_jev.py` — `200` golden vs `Jev` (`$1`, `TYPESAFE_API_KEY`)
* `laya/train.py` — `RTX2060 12GB` single-GPU, offline (`head 256`, `batch 8` `~3.5GB`)

**Quickstart (on RTX2060):** `BOUNDARY.md` §5 copy-paste.

**Fine-tuned output:** `laya/laya_biome_finetuned/model.safetensors` (`804M`) — load via `laya.load("laya/laya_biome_finetuned")` or `LayaGateway(model="...")` with `seeded` `System.Random` for deterministic variety.
