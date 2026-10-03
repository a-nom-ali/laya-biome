# data — Generated, not committed (except raw_states.json example)

- `raw_states.json` — `C#` `HeightmapGen` export: `[{latitude, altitude, distance_to_ocean, landmass_size, ocean_current, windward, rainfall}, ...]` (`5k` `16×16` chunks, oversampled boundaries)
- `laya_terrain.jsonl` — `scripts/label_with_nimble.py` output: `[{task, state, questions, gold}, ...]` (`~10k` entries, 2 tasks per raw)
- `train.jsonl` / `test.jsonl` — `80/20` split of `laya_terrain.jsonl` for `laya/train.py`

All are `gitignore`'d except the `5k` example `raw_states.json` (keep it small).
