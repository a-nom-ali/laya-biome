using System;
using System.Collections.Generic;

// Discrete per-biome per-tile LOD palettes — keeps choice ≤12 (head 256/12=21 tokens/option)
// lod 0=planet coarsest, 4=ground finest. Per-tile seed ensures variety but deterministic.
// RTX2060 12GB: fine, 421M + 1.3KB/chunk cache in 64GB RAM.

public static class Palettes
{
    // Returns palette options for (terrain, lod). Never >12; global would be 60 → use per-biome.
    public static string[] For(string terrain, int lod)
    {
        lod = Math.Clamp(lod, 0, 4);
        return (terrain, lod) switch
        {
            ("Plains", 0) => new[] { "Steppe Tan" },
            ("Plains", 1) => new[] { "Steppe Tan", "Dry Grass" },
            ("Plains", 2) => new[] { "Grass Tufts", "Soil Cracks", "Dry Grass" },
            ("Plains", 3) => new[] { "Grass Tufts", "Soil Cracks", "Wildflowers", "Bare Dirt" },
            ("Plains", 4) => new[] { "Grass Tufts", "Soil Cracks", "Wildflowers", "Bare Dirt", "Pebbles" },

            ("Hills", 0) => new[] { "Hill Wash" },
            ("Hills", 1) => new[] { "Hill Wash", "Scrub Haze" },
            ("Hills", 2) => new[] { "Deciduous Scrub", "Grassland Hills", "Rock Outcrop" },
            ("Hills", 3) => new[] { "Deciduous Scrub", "Pine Scrub", "Grassland Hills", "Rock Outcrop" },
            ("Hills", 4) => new[] { "Deciduous Scrub", "Pine Scrub", "Grassland Hills", "Rock Outcrop", "Bare Stone" },

            ("Mountains", 0) => new[] { "Alpine Gray" },
            ("Mountains", 1) => new[] { "Alpine Gray", "Snow Cap" },
            ("Mountains", 2) => new[] { "Granite", "Scree", "Snow Cap" },
            ("Mountains", 3) => new[] { "Granite", "Scree", "Lichen Patch", "Snow Cap" },
            ("Mountains", 4) => new[] { "Granite", "Scree", "Lichen Patch", "Snow Cap", "Bare Rock" },

            ("Canyons", 0) => new[] { "Canyon Wash" },
            ("Canyons", 1) => new[] { "Canyon Wash", "Red Walls" },
            ("Canyons", 2) => new[] { "Desert Scrub", "Bare Sandstone", "Red Walls" },
            ("Canyons", 3) => new[] { "Desert Scrub", "Riparian Cottonwood", "Bare Sandstone", "Red Walls" },
            ("Canyons", 4) => new[] { "Desert Scrub", "Riparian Cottonwood", "Bare Sandstone", "Red Walls", "Salt Crust" },

            ("Dunes", 0) => new[] { "Sand Seas" },
            ("Dunes", 1) => new[] { "Sand Seas", "Dune Shadows" },
            ("Dunes", 2) => new[] { "Dune Sand", "Wind Ripples", "Dune Shadows" },
            ("Dunes", 3) => new[] { "Dune Sand", "Wind Ripples", "Desert Varnish", "Dune Shadows" },
            ("Dunes", 4) => new[] { "Dune Sand", "Wind Ripples", "Desert Varnish", "Salt Flat", "Bare Sand" },

            _ => new[] { "Base" }
        };
    }

    // Seeded-deterministic per tile: sample from laya's probabilities, then cache per (x,y,lod)
    // Usage: int idx = Palettes.SeededChoice(probs, HashCode.Combine(planetSeed, x, y, lod));
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

    // Helper: hash planetSeed + tile + lod for deterministic variety
    public static int TileSeed(int planetSeed, int x, int y, int lod) => HashCode.Combine(planetSeed, x, y, lod);
}
