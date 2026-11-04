"""Command line: umwelt data | run | report | report-v2 | atlas."""

import argparse

from umwelt import learner
from umwelt.presets import PRESETS
from umwelt.world import WORLDS

ALL_WORLDS = list(WORLDS) + ["lawful_mj", "impermanent_mj", "ghost_mj"]


def main():
    ap = argparse.ArgumentParser(prog="umwelt")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="generate recorded worlds")
    d.add_argument("--worlds", nargs="+", default=list(WORLDS), choices=ALL_WORLDS)
    d.add_argument("--episodes", type=int, default=1280)

    r = sub.add_parser("run", help="train every mind in a world and run all tests")
    r.add_argument("--world", default="lawful", choices=ALL_WORLDS)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--presets", nargs="+", default=list(PRESETS))
    r.add_argument("--steps", type=int, default=learner.LearnerConfig.steps)
    r.add_argument("--horizons", type=int, nargs="+", default=list(learner.LearnerConfig.horizons))
    r.add_argument("--obs-dropout", type=float, default=learner.LearnerConfig.obs_dropout)
    r.add_argument("--objective", default="pixel", choices=["pixel", "jepa"])
    r.add_argument("--no-controls", action="store_true")
    r.add_argument("--tag", default="main")
    r.add_argument("--data-seed", type=int, default=0, help="which recording of the world to train and test on")
    r.add_argument("--n-train", type=int, default=1024, help="training episodes (Amendment 4: 8192)")

    rp = sub.add_parser("report", help="aggregate results into tables, figures and REPORT.md")
    rp.add_argument("--tag", default="main")

    r2 = sub.add_parser("report-v2", help="check Amendment 2 (the memory world) and write results/report_v2")
    r2.add_argument("--world", default="occlusion")

    sub.add_parser("atlas", help="enumerate coherent positions on classic aporias")

    args = ap.parse_args()
    if args.cmd == "data":
        from umwelt.data import save

        for w in args.worlds:
            save(w, args.episodes)
    elif args.cmd == "run":
        from umwelt.experiment import run_world

        cfg = learner.LearnerConfig(steps=args.steps, horizons=tuple(args.horizons), obs_dropout=args.obs_dropout,
                                    objective=args.objective)
        run_world(args.world, args.seed, args.presets, cfg, controls=not args.no_controls, tag=args.tag,
                  data_seed=args.data_seed, n_train=args.n_train)
    elif args.cmd == "atlas":
        from umwelt.atlas import build as build_atlas

        build_atlas()
    elif args.cmd == "report":
        from umwelt.report import build

        build(args.tag)
    elif args.cmd == "report-v2":
        from umwelt.report_v2 import build as build_v2

        build_v2(args.world)


if __name__ == "__main__":
    main()
