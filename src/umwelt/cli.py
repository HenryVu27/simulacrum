"""Command line: umwelt data | run | report."""

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
    r.add_argument("--no-controls", action="store_true")
    r.add_argument("--tag", default="main")

    rp = sub.add_parser("report", help="aggregate results into tables, figures and REPORT.md")
    rp.add_argument("--tag", default="main")

    args = ap.parse_args()
    if args.cmd == "data":
        from umwelt.data import save

        for w in args.worlds:
            save(w, args.episodes)
    elif args.cmd == "run":
        from umwelt.experiment import run_world

        cfg = learner.LearnerConfig(steps=args.steps)
        run_world(args.world, args.seed, args.presets, cfg, controls=not args.no_controls, tag=args.tag)
    elif args.cmd == "report":
        from umwelt.report import build

        build(args.tag)


if __name__ == "__main__":
    main()
