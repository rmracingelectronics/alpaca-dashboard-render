from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Convenience wrapper: build dataset, train Q-learning policy, and run an out-of-sample backtest.")
    p.add_argument("--source-csv", default="", help="Use an existing all-candidates CSV instead of rebuilding from bars.")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-06-20")
    p.add_argument("--train-end", required=True)
    p.add_argument("--validate-end", required=True)
    p.add_argument("--preset", default="v25_playbook")
    p.add_argument("--symbols", default="")
    p.add_argument("--feed", default="iex")
    p.add_argument("--session-mode", default="regular_only")
    p.add_argument("--top-trades-per-day", type=int, default=1)
    p.add_argument("--min-edge", type=float, default=0.0)
    p.add_argument("--min-state-count", type=int, default=8)
    p.add_argument("--kappa", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--name", default="")
    args = p.parse_args()

    py = sys.executable
    dataset_name = args.name + "_dataset" if args.name else ""
    build_cmd = [py, "tools/build_q_learning_dataset.py", "--start", args.start, "--end", args.end, "--preset", args.preset, "--feed", args.feed, "--session-mode", args.session_mode]
    if args.symbols:
        build_cmd += ["--symbols", args.symbols]
    if args.source_csv:
        build_cmd += ["--source-csv", args.source_csv]
    if dataset_name:
        build_cmd += ["--name", dataset_name]
    run(build_cmd)

    # Locate the most recent matching dataset folder.
    import glob
    candidates = sorted(glob.glob("data/ml_datasets/q_learning_candidates_*/candidates.csv") + glob.glob(f"data/ml_datasets/{dataset_name}/candidates.csv"), key=lambda p: Path(p).stat().st_mtime)
    if not candidates:
        raise RuntimeError("Could not locate generated candidates.csv")
    dataset = candidates[-1]
    model_name = args.name + "_model" if args.name else ""
    train_cmd = [py, "tools/train_q_learning_policy.py", "--dataset", dataset, "--train-end", args.train_end, "--validate-end", args.validate_end, "--top-trades-per-day", str(args.top_trades_per_day), "--min-edge", str(args.min_edge), "--min-state-count", str(args.min_state_count), "--kappa", str(args.kappa), "--epochs", str(args.epochs)]
    if model_name:
        train_cmd += ["--name", model_name]
    run(train_cmd)

    models = sorted(glob.glob("data/ml_models/q_policy_*/model.json") + glob.glob(f"data/ml_models/{model_name}/model.json"), key=lambda p: Path(p).stat().st_mtime)
    if not models:
        raise RuntimeError("Could not locate generated model.json")
    model = models[-1]
    run([py, "tools/backtest_q_learning_policy.py", "--dataset", dataset, "--model", model, "--start", args.validate_end, "--end", args.end, "--top-trades-per-day", str(args.top_trades_per_day), "--min-edge", str(args.min_edge), "--min-state-count", str(args.min_state_count), "--name", (args.name + "_final_backtest" if args.name else "")])


if __name__ == "__main__":
    main()
