# data — Generated, not committed (except raw_states.json example)

- `raw_states.json` — `C#` `HeightmapGen` export: `[{latitude, altitude, distance_to_ocean, landmass_size, ocean_current, windward, rainfall, lod, lod_blend}, ...]` (`5k` `16×16` chunks `×5` `LOD` `0..4` → `25k` `9-field` states, per-tile `hash(planet_seed,x,y,lod)`, oversampled `rainfall 0.45-0.55` + `palette` boundaries)
- `laya_terrain.jsonl` — `scripts/label_with_nimble.py` output: `[{task, state, questions, gold}, ...]` (`~15k` entries, `3` tasks per raw: `macro` + `micro` + `palette` `≤12` + `blend` `0..3`, discrete per-biome per-tile)
- `train.jsonl` / `test.jsonl` — `80/20` split of `laya_terrain.jsonl` for `laya/train.py` (`single-GPU` `RTX2060` `8×1024` `~3.5GB`)

All are `gitignore`'d except the `5k` example `raw_states.json` (keep it small).
