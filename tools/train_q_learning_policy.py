from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import pandas as pd

from src.config import ML_MODELS_DIR
from src.q_learning_policy import (
    QLearningPolicyConfig,
    backtest_q_policy,
    save_q_model,
    split_by_dates,
    train_q_learning_policy,
)


def _dt_stamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    p = argparse.ArgumentParser(description="Train a tabular Q-learning trade/skip policy on live-safe candidate rows.")
    p.add_argument("--dataset", required=True, help="Path to candidates.csv produced by build_q_learning_dataset.py")
    p.add_argument("--train-end", required=True, help="Last session date used for training, e.g. 2024-12-31")
    p.add_argument("--validate-end", required=True, help="Last session date used for validation; later rows are test.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument("--kappa", type=float, default=0.25)
    p.add_argument("--reward-mu", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--min-state-count", type=int, default=8)
    p.add_argument("--min-edge", type=float, default=0.0)
    p.add_argument("--top-trades-per-day", type=int, default=1)
    p.add_argument("--max-symbol-per-day", type=int, default=1)
    p.add_argument("--fixed-risk", type=float, default=100.0)
    p.add_argument("--name", default="")
    args = p.parse_args()

    df = pd.read_csv(args.dataset)
    if df.empty:
        raise RuntimeError("Dataset is empty.")
    df = split_by_dates(df, train_end=args.train_end, validate_end=args.validate_end)
    train = df[df["split"] == "train"].copy()
    validate = df[df["split"] == "validate"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty:
        raise RuntimeError("Training split is empty. Check --train-end.")

    cfg = QLearningPolicyConfig(
        alpha=float(args.alpha), gamma=float(args.gamma), epsilon=float(args.epsilon),
        kappa=float(args.kappa), reward_mu=float(args.reward_mu), train_epochs=int(args.epochs),
        min_state_count=int(args.min_state_count), min_edge=float(args.min_edge),
    )
    model = train_q_learning_policy(train, cfg)
    model["dataset_path"] = str(Path(args.dataset).resolve())
    model["splits"] = {
        "train_rows": int(len(train)), "validate_rows": int(len(validate)), "test_rows": int(len(test)),
        "train_end": str(args.train_end), "validate_end": str(args.validate_end),
    }
    name = args.name or f"q_policy_{Path(args.dataset).parent.name}_{_dt_stamp()}"
    out_dir = ML_MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_q_model(model, out_dir / "model.json")

    summaries = []
    for split_name, part in [("train", train), ("validate", validate), ("test", test)]:
        selected, summary, reviewed = backtest_q_policy(
            part, model,
            top_trades_per_day=int(args.top_trades_per_day),
            max_symbol_per_day=int(args.max_symbol_per_day),
            min_edge=float(args.min_edge),
            min_state_count=int(args.min_state_count),
            fixed_risk_dollars=float(args.fixed_risk),
        )
        summary["split"] = split_name
        summaries.append(summary)
        selected.to_csv(out_dir / f"{split_name}_selected_trades.csv", index=False)
        reviewed.to_csv(out_dir / f"{split_name}_reviewed_candidates.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "split_summary.csv", index=False)
    pd.DataFrame(model.get("state_stats", [])).sort_values("q_edge", ascending=False).to_csv(out_dir / "q_state_table.csv", index=False)
    (out_dir / "training_config.json").write_text(json.dumps({"args": vars(args), "config": cfg.__dict__}, indent=2), encoding="utf-8")
    print(f"Model saved: {model_path}")
    print(f"Output folder: {out_dir}")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
