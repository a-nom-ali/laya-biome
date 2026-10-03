using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

// Demo: generates 5k 16x16 chunks with Chamfer + per-continent + wind-sweep, oversampled, to data/raw_states.json
// Run: dotnet run --project csharp/laya-biome.csproj

class Program
{
    static void Main()
    {
        Console.WriteLine("Laya-Biome HeightmapGen — generating raw_states.json (Chamfer, per-continent, 5 LODs)");
        int width = 16, height = 16;
        int chunks = 5000;
        var allStates = new List<Dictionary<string, object>>();
        var rng = new Random(42);

        for (int c = 0; c < chunks; c++)
        {
            // Random heightmap and water mask for demo (replace with real heightmap in your engine)
            float[] heightMap = new float[width * height];
            bool[] isWater = new bool[width * height];
            for (int i = 0; i < width * height; i++)
            {
                heightMap[i] = (float)rng.NextDouble();
                isWater[i] = heightMap[i] < 0.3; // 30% water
            }

            // Pre-passes (O(N) on float[])
            float[] dist = HeightmapGen.CalculateDistanceToOcean(isWater, width, height, useChamfer: true);
            float[] landSize = HeightmapGen.CalculateLandmassSize(isWater, width, height);
            // Wind-sweep: Westerlies for y < height/2, Trade Winds otherwise (demo)
            float[] rain = HeightmapGen.CalculateRainShadows(heightMap, isWater, width, height, y => y < height / 2);

            // Sample random tiles, oversample boundaries (rainfall 0.45-0.55, windward 0.1/dist 0.8)
            for (int s = 0; s < 1; s++) // 1 sample per chunk for demo; increase for more
            {
                int x = rng.Next(width), y = rng.Next(height);
                int idx = y * width + x;
                float rainfall = rain[idx];
                float windward = (x > 0) ? (heightMap[idx] > heightMap[idx - 1] ? 0.8f : 0.2f) : 0.5f;

                // Oversample edge cases
                bool isEdge = (rainfall > 0.45 && rainfall < 0.55) || (windward < 0.2 && dist[idx] > 0.7);
                if (!isEdge && rng.NextDouble() < 0.7) continue; // keep 30% of non-edge

                // Emit 5 LODs per tile (discrete 0..4)
                for (int lod = 0; lod < 5; lod++)
                {
                    var p = new BiomeParams
                    {
                        latitude = (float)(rng.NextDouble() * 180 - 90),
                        altitude = heightMap[idx],
                        distanceToOcean = dist[idx],
                        landmassSize = landSize[idx],
                        oceanCurrent = (float)(rng.NextDouble() * 2 - 1),
                        windward = windward,
                        rainfall = rainfall,
                        lod = lod,
                        lodBlend = (float)rng.NextDouble()
                    };
                    var state = BiomeStateBuilder.ToLayaState(p, seed: rng.Next());
                    // Also add lod for training
                    state["lod"] = lod;
                    state["lod_blend"] = p.lodBlend;
                    allStates.Add(state);
                    if (allStates.Count >= 25000) break;
                }
                if (allStates.Count >= 25000) break;
            }
            if (c % 500 == 0) Console.WriteLine($"  chunk {c}/{chunks} → {allStates.Count} states");
            if (allStates.Count >= 25000) break;
        }

        string outPath = "data/raw_states.json";
        Directory.CreateDirectory("data");
        var json = JsonSerializer.Serialize(allStates, new JsonSerializerOptions { WriteIndented = false });
        File.WriteAllText(outPath, json);
        Console.WriteLine($"Wrote {allStates.Count} states to {outPath} ({new FileInfo(outPath).Length} bytes)");
        Console.WriteLine("Next: python scripts/label_with_nimble.py --input data/raw_states.json --output data/laya_terrain.jsonl");
    }
}
