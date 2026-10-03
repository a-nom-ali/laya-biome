using System;
using System.Collections.Generic;

// Laya-Biome: Planet-curved heightmap pre-pass
// - Chamfer distance (1 / 1.414) for coastal proximity (great-circle approx, not Manhattan)
// - Per-continent landmass labeling (flood-fill, not global water %)
// - Directional wind-sweep for rain shadows (per latitude band)
// All O(N) on flat float[] (y*width+x), 1D for CPU cache contiguity.
// RTX2060 12GB: fine for 2048^2; use Compute Shader Jump Flooding beyond.

public static class HeightmapGen
{
    // Distance to ocean: Chamfer 3x3 (8-neighbour) 2-pass. Returns normalized 0..1 (0=coastal, 1=deep interior).
    public static float[] CalculateDistanceToOcean(bool[] isWater, int width, int height, bool useChamfer = true)
    {
        int n = width * height;
        float[] dist = new float[n];
        float maxDist = width + height; // safe upper bound
        float diag = 1.41421356f;

        // Pass 1: TL -> BR
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;
                if (isWater[i]) { dist[i] = 0f; continue; }
                float best = maxDist;
                if (x > 0) best = Math.Min(best, dist[y * width + (x - 1)] + 1f);
                if (y > 0) best = Math.Min(best, dist[(y - 1) * width + x] + 1f);
                if (useChamfer)
                {
                    if (x > 0 && y > 0) best = Math.Min(best, dist[(y - 1) * width + (x - 1)] + diag);
                    if (x < width - 1 && y > 0) best = Math.Min(best, dist[(y - 1) * width + (x + 1)] + diag);
                }
                dist[i] = best;
            }
        }
        // Pass 2: BR -> TL
        for (int y = height - 1; y >= 0; y--)
        {
            for (int x = width - 1; x >= 0; x--)
            {
                int i = y * width + x;
                if (isWater[i]) continue;
                float best = dist[i];
                if (x < width - 1) best = Math.Min(best, dist[y * width + (x + 1)] + 1f);
                if (y < height - 1) best = Math.Min(best, dist[(y + 1) * width + x] + 1f);
                if (useChamfer)
                {
                    if (x < width - 1 && y < height - 1) best = Math.Min(best, dist[(y + 1) * width + (x + 1)] + diag);
                    if (x > 0 && y < height - 1) best = Math.Min(best, dist[(y + 1) * width + (x - 1)] + diag);
                }
                dist[i] = best;
            }
        }
        // Normalize 0..1 by max (planet great-circle approx = pi*R, here use observed max)
        float maxObs = 0f;
        for (int i = 0; i < n; i++) if (!isWater[i] && dist[i] < maxDist) maxObs = Math.Max(maxObs, dist[i]);
        if (maxObs > 0f) for (int i = 0; i < n; i++) if (!isWater[i]) dist[i] /= maxObs;
        // Water stays 0
        return dist;
    }

    // Per-continent landmass size: flood-fill labeling. Returns 0..1 per tile (0=water, 1=Pangea).
    public static float[] CalculateLandmassSize(bool[] isWater, int width, int height)
    {
        int n = width * height;
        int[] label = new int[n];
        for (int i = 0; i < n; i++) label[i] = -1;
        int nextId = 0;
        var sizes = new List<int>();
        var stack = new Stack<int>();

        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int i = y * width + x;
            if (isWater[i] || label[i] != -1) continue;
            // BFS new continent
            int count = 0;
            stack.Push(i); label[i] = nextId;
            while (stack.Count > 0)
            {
                int cur = stack.Pop(); count++;
                int cx = cur % width, cy = cur / width;
                // 4-neighbour (Chamfer already handles diagonals for distance; keep 4 for contiguity)
                if (cx > 0) { int nb = cur - 1; if (!isWater[nb] && label[nb]==-1) { label[nb]=nextId; stack.Push(nb); } }
                if (cx < width-1) { int nb = cur + 1; if (!isWater[nb] && label[nb]==-1) { label[nb]=nextId; stack.Push(nb); } }
                if (cy > 0) { int nb = cur - width; if (!isWater[nb] && label[nb]==-1) { label[nb]=nextId; stack.Push(nb); } }
                if (cy < height-1) { int nb = cur + width; if (!isWater[nb] && label[nb]==-1) { label[nb]=nextId; stack.Push(nb); } }
            }
            sizes.Add(count);
            nextId++;
        }
        int maxSize = 0; foreach (var s in sizes) maxSize = Math.Max(maxSize, s);
        float[] out_ = new float[n];
        for (int i = 0; i < n; i++)
        {
            if (isWater[i]) out_[i] = 0f;
            else out_[i] = maxSize > 0 ? (float)sizes[label[i]] / maxSize : 0f;
        }
        return out_;
    }

    // Rain shadows: directional wind-sweep per latitude band. Returns 0..1 rainfall.
    // isWater: bool[], heightMap: 0..1, width/height, latitude per row is handled outside (y -> lat).
    public static float[] CalculateRainShadows(float[] heightMap, bool[] isWater, int width, int height, Func<int, bool> isWestToEast)
    {
        float[] rain = new float[width * height];
        float baseMoisture = 1.0f, recharge = 0.05f, lift = 3.5f, baseline = 0.05f;

        for (int y = 0; y < height; y++)
        {
            bool w2e = isWestToEast(y); // true = Westerlies (W->E), false = Trade Winds (E->W)
            float wind = baseMoisture;
            // Sweep direction
            if (w2e)
            {
                for (int x = 0; x < width; x++)
                {
                    int i = y * width + x;
                    float r = 0f;
                    if (isWater[i]) { wind = Math.Min(1f, wind + recharge); r = wind * baseline; }
                    else
                    {
                        if (x > 0)
                        {
                            float dh = heightMap[i] - heightMap[i - 1];
                            if (dh > 0) { float sq = dh * lift; r = Math.Min(wind, sq); wind = Math.Max(0f, wind - r); r += wind * baseline; }
                            else r = wind * baseline;
                        }
                        else r = wind * baseline;
                    }
                    rain[i] = r;
                }
            }
            else
            {
                float wind2 = baseMoisture;
                for (int x = width - 1; x >= 0; x--)
                {
                    int i = y * width + x;
                    float r = 0f;
                    if (isWater[i]) { wind2 = Math.Min(1f, wind2 + recharge); r = wind2 * baseline; }
                    else
                    {
                        if (x < width - 1)
                        {
                            float dh = heightMap[i] - heightMap[i + 1];
                            if (dh > 0) { float sq = dh * lift; r = Math.Min(wind2, sq); wind2 = Math.Max(0f, wind2 - r); r += wind2 * baseline; }
                            else r = wind2 * baseline;
                        }
                        else r = wind2 * baseline;
                    }
                    rain[i] = r;
                }
            }
        }
        // Normalize 0..1
        float maxR = 0f; foreach (var v in rain) maxR = Math.Max(maxR, v);
        if (maxR > 0f) for (int i = 0; i < rain.Length; i++) rain[i] /= maxR;
        return rain;
    }
}
