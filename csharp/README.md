# csharp — Heightmap Pre-pass + LOD Palettes

Builds the `9-field` `BiomeParams` (`+ lod 0..4` + `lodBlend`) for `laya` *before* any model call.

* `HeightmapGen.cs:CalculateDistanceToOcean(isWater, w, h)` — **Chamfer** `2-pass` (`1` / `1.414`), normalized `0..1` (planet great-circle)
* `HeightmapGen.cs:CalculateLandmassSize(isWater, w, h)` — **per-continent** `flood-fill` `0..1` (`1=Pangea`, not global water %)
* `HeightmapGen.cs:CalculateRainShadows(...)` — per-latitude `W→E` / `E→W` sweep
* `Palettes.cs:For(terrain, lod)` — **discrete per-biome per-tile** `≤12` per `(terrain,lod)` (`Dunes@0 → Sand Seas` vs `@4 → Dune Sand/Wind Ripples/...` `4`), `SeededChoice` per tile `hash(planetSeed,x,y,lod)`

All `O(N)` on `float[]` (`y*width+x`), `~3ms` for `1024²` on CPU, `Jump Flooding` on `Compute Shader` for `2048²`. `RTX2060 12GB` holds `laya` `3.5GB` + `1.3KB/chunk` cache in `64GB RAM`.
