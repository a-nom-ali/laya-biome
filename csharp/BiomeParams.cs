using System;

// Typed state for Laya — mirrors engine/state.py but for terrain.
// All floats normalized 0..1 or -1..1, sent as System One state.

public struct BiomeParams
{
    public float latitude;          // -90..90 -> normalized -1..1 for laya
    public float altitude;          // heightMap 0..1
    public float distanceToOcean;   // Chamfer 0..1 (0=coastal)
    public float landmassSize;      // per-continent 0..1 (1=Pangea)
    public float oceanCurrent;      // -1 cold polar -> +1 warm equatorial
    public float windward;          // 0 leeward (shadow) .. 1 windward
    public float rainfall;          // 0..1 from CalculateRainShadows
}

public static class BiomeStateBuilder
{
    public static Dictionary<string, object> ToLayaState(BiomeParams p, int seed = 0)
    {
        return new Dictionary<string, object>
        {
            { "latitude", p.latitude },
            { "altitude", p.altitude },
            { "distance_to_ocean", p.distanceToOcean },
            { "landmass_size", p.landmassSize },
            { "ocean_current", p.oceanCurrent },
            { "windward", p.windward },
            { "rainfall", p.rainfall },
            { "seed", seed } // for sampled-deterministic cache key
        };
    }

    // Seeded-deterministic sampling: hash(planet_seed, x, y, lod) -> categorical
    public static int SeededChoice(float[] probs, int seed)
    {
        var rng = new System.Random(seed);
        float r = (float)rng.NextDouble();
        float acc = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            acc += probs[i];
            if (r <= acc) return i;
        }
        return probs.Length - 1;
    }
}
