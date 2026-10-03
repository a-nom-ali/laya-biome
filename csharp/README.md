# csharp — Heightmap Pre-pass

Builds the 7-field `BiomeParams` for `laya` *before* any model call (engine calculates, model decides).

* `HeightmapGen.cs:CalculateDistanceToOcean(isWater, w, h)` — Chamfer `2-pass` (`1` / `1.414`), normalized `0..1`
* `HeightmapGen.cs:CalculateLandmassSize(isWater, w, h)` — `flood-fill` per-continent `0..1` (`1=Pangea`)
* `HeightmapGen.cs:CalculateRainShadows(heightMap, isWater, w, h, isWestToEast)` — per-latitude wind direction

All `O(N)` on `float[]` (`y*width+x`), `~3ms` for `1024²` on CPU, `Jump Flooding` on `Compute Shader` for `2048²`.
