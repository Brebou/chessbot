"""Run evaluation of model vs engine (replicates evaluate.py).

This is a wrapper around chess_hw.match.evaluate_vs_engine for consistency.

Usage:
    uv run diag/run_eval.py --model base_model --seeds 0 1 --levels 0 1 --n_openings 10 --workers 4
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw.match import evaluate_vs_engine


def main():
    parser = argparse.ArgumentParser(description="Evaluate model vs engine")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds to use")
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3], help="Engine levels to evaluate against")
    parser.add_argument("--n_openings", type=int, default=50, help="Number of openings per seed")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--out", type=str, default=None, help="Output directory for results")
    parser.add_argument("--games_out", action="store_true", help="Save individual games")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to chess-post-training
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / args.model
    
    results = {}
    
    for seed in args.seeds:
        print(f"\nEvaluating with seed {seed}...")
        summary = evaluate_vs_engine(
            model_path=str(model_path),
            levels=args.levels,
            n_openings=args.n_openings,
            seed=seed,
            workers=args.workers,
            progress=True,
            games_out=f"{args.out}/seed{seed}.games.jsonl" if args.out and args.games_out else None
        )
        results[f"seed_{seed}"] = summary
        
        # Print summary
        print(f"\nSeed {seed} results:")
        print(f"  Mean score: {summary['mean_score']:.3f}")
        print(f"  Legal first try: {summary['legal_first_try']*100:.1f}%")
        print(f"  Legal within 3: {summary['legal_within_3']*100:.1f}%")
        print(f"  Random move rate: {summary['random_move_rate']*100:.1f}%")
        for level, data in summary['per_level'].items():
            print(f"  Level {level}: W={data['wins']} D={data['draws']} L={data['losses']} Score={data['score']:.3f}")
    
    # Save results
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_path = out_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {summary_path}")
    
    return results


if __name__ == "__main__":
    main()
