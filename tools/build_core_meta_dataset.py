from __future__ import annotations
import argparse, json
from src.meta_label import build_core_meta_dataset


def main():
    p = argparse.ArgumentParser(description="Build V24 core meta-label dataset from verified core rule triggers.")
    p.add_argument("--preset", default="edge_core_40")
    p.add_argument("--symbols", default="", help="Comma-separated custom symbols; overrides preset if supplied.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--feed", default="iex")
    p.add_argument("--min-score", type=float, default=80.0)
    p.add_argument("--direction", default="long_only", choices=["long_only", "short_only", "long_short"])
    p.add_argument("--target-r-values", default="0.5,0.6,0.75,1.0,1.5")
    p.add_argument("--horizon-bars", type=int, default=12)
    p.add_argument("--split-train-end", default="2024-12-31")
    p.add_argument("--split-validate-end", default="2025-12-31")
    args = p.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    targets = [float(x) for x in args.target_r_values.replace(";", ",").split(",") if x.strip()]
    result = build_core_meta_dataset(
        preset=args.preset, symbols=symbols, start=args.start, end=args.end, feed=args.feed,
        min_score=args.min_score, direction_mode=args.direction, target_r_values=targets,
        horizon_bars=args.horizon_bars, split_train_end=args.split_train_end, split_validate_end=args.split_validate_end,
    )
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
