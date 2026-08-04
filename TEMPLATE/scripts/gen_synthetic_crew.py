"""Generate a synthetic crew JSON for testing/demos.

    python scripts/gen_synthetic_crew.py --n 6 --seed 3 --out data/crew/synth_crew.json

Uses the registry's latent traits so generated crews always match the current
variable registry, and Faker for plausible names.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from nfp.registry import load_registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--registry", default="data/registry/registry_sample.csv")
    ap.add_argument("--out", default="data/crew/synth_crew.json")
    args = ap.parse_args()

    try:
        from faker import Faker
        fake = Faker()
        Faker.seed(args.seed)
        name = lambda: f"{fake.first_name()[0]}. {fake.last_name()}"
    except ImportError:
        name = lambda: "Crew Member"

    rng = np.random.default_rng(args.seed)
    traits = load_registry(args.registry).trait_codes

    crew = []
    for i in range(args.n):
        crew.append({
            "id": f"S{i + 1:03d}",
            "name": name(),
            "traits": {
                code: {
                    "mean": round(float(rng.normal(0, 0.8)), 2),
                    "sd": round(float(rng.uniform(0.2, 0.5)), 2),
                }
                for code in traits
            },
        })

    out = Path(args.out)
    out.write_text(json.dumps({"crew": crew}, indent=2), encoding="utf-8")
    print(f"Wrote {args.n} synthetic candidates -> {out}")


if __name__ == "__main__":
    main()
