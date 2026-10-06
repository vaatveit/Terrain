"""Tune the mineral weights so each deposit type carries its intended share of a resource.

A sim samples a resource field in proportion to score times AREA, so a weight is not a
share: a type that covers whole provinces outweighs one confined to a few river reaches
whatever the two are weighted. This measures how much ground each type covers on generated
worlds and prints the weights under which each carries the share in SHARES below. The
weights in `minerals.MineralParams` are this script's output.

Give it PRODUCTION worlds, n=785. The routed types cannot be measured on anything smaller --
the reach of a river is a world-scale length and a river is a cell-scale thing, so placer
gold covers 20x less ground per unit of land at n=128 -- and neither can the climatic ones,
which came out 2x off between n=128 and n=785 on the same seed.

Run:  .venv\\Scripts\\python.exe scripts\\tune_mineral_weights.py data/world_a data/world_b
"""

import argparse
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import export, minerals  # noqa: E402

from generate_world import load_or_build  # noqa: E402

# What each deposit type should carry. Copper's are sourced: porphyry deposits are ~60% of
# world production, split here as 40 unenriched and 20 enriched, and sediment-hosted ~23%.
# The rest is a judgement the sources only order:
#   gold  orogenic first; placers are 15,500 t against 22,000 t of lode gold in the younger
#         belts and ~24,000 t in the shields, so about a quarter of that family.
#   iron  high-grade ore in iron formation is the bulk of production, so enrichment leads
#         the formation it enriches; Kiruna-type and channel iron are minor.
SHARES = {
    "copper": {"porphyry": 0.40, "supergene_copper": 0.20, "sediment_copper": 0.23,
               "iocg": 0.17},
    "gold": {"orogenic_gold": 0.40, "placer_gold": 0.15, "epithermal_hs": 0.15,
             "epithermal_ls": 0.15, "porphyry": 0.10, "iocg": 0.05},
    "iron": {"enriched_iron": 0.50, "bif_iron": 0.35, "kiruna_iron": 0.10,
             "channel_iron": 0.05},
}

FIELDS = ("land_mask", "is_continental", "boundary_stress", "boundary_speed",
          "boundary_slip", "boundary_dist_km", "nearest_boundary", "uplift",
          "precipitation_mm", "temperature_c", "flow_receiver", "slope", "river_width_m")


def type_mass(world_dir):
    """Each deposit type's score summed over area, per unit of the world's land area."""
    d = np.load(pathlib.Path(world_dir) / "world.npz", allow_pickle=False)
    grid = load_or_build(int(d["_n"]))
    _, fields, meta = export.load_native(pathlib.Path(world_dir) / "world.npz", grid=grid)
    cratons = minerals.invent_cratons(grid, meta["seed"], fields["is_continental"])
    scores = minerals.deposit_scores(
        radius_km=grid.radius_km, cell_km=float(grid.arc_dist[grid.neighbours >= 0].mean()),
        craton_margin_km=cratons["craton_margin_km"],
        iron_formation=cratons["iron_formation"], **{k: fields[k] for k in FIELDS})
    land = grid.area[fields["land_mask"].astype(bool)].sum()
    return meta, {t: float((grid.area * s).sum() / land) for t, s in scores.items()}


def carried(mass, weights):
    total = sum(weights[t] * mass[t] for t in weights)
    return {t: weights[t] * mass[t] / total for t in weights}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("worlds", nargs="+", help="world directories holding world.npz")
    args = ap.parse_args()

    runs = []
    for world in args.worlds:
        meta, mass = type_mass(world)
        runs.append((f"n={meta['n']} s={meta['seed']}", mass))
        print(f"measured {world}  ({runs[-1][0]})")
    mean = {t: float(np.mean([m[t] for _, m in runs])) for t in runs[0][1]}

    now = minerals.MineralParams()
    labels = "  ".join(f"{name:>12s}" for name, _ in runs)
    for resource, shares in SHARES.items():
        tuned = minerals.weights_for_shares({t: mean[t] for t in shares}, shares)
        # Floored, so the printed weights still sum to at most 1.
        tuned = {t: math.floor(w * 1e4) / 1e4 for t, w in tuned.items()}
        current = getattr(now, resource)
        print(f"\n{resource}: share carried by each type, per world")
        print(f"  {'type':18s} {'target':>6s}   {'weight':>6s}  {labels}   ->   "
              f"{'weight':>6s}  {labels}")
        before = [carried(m, current) for _, m in runs]
        after = [carried(m, tuned) for _, m in runs]
        for t in shares:
            print(f"  {t:18s} {shares[t]:6.0%}   {current.get(t, 0.0):6.3f}  "
                  + "  ".join(f"{b.get(t, 0.0):12.1%}" for b in before)
                  + f"   ->   {tuned[t]:6.4f}  " + "  ".join(f"{a[t]:12.1%}" for a in after))
        print(f"  {resource}: dict = dataclasses.field(default_factory=lambda: {tuned})")


if __name__ == "__main__":
    main()
