"""Analyze games to extract metrics for C4, C8, C11, C12.

This module analyzes game records to compute:
- C4: Force brute (score vs engine by level)
- C8: Conversion de l'avantage (win rate from +2/+3/+5/+9)
- C11: Shuffling/repetition analysis
- C12: Variance and robustness

Usage:
    uv run diag/analyze_games.py --games results/eval_base --out results/analysis
    uv run diag/analyze_games.py --games results/eval_base/seed0.games.jsonl --out results/analysis
    uv run diag/analyze_games.py --games 'results/eval_base/*.games.jsonl' --out results/analysis
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import chess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .common import material_diff, jsonl_read
except ImportError:
    from diag.common import material_diff, jsonl_read


CONVERSION_THRESHOLDS = (2, 3, 5, 9)


@dataclass
class GameAnalysis:
    """Analysis results for a single game."""
    game_id: str
    moves: List[str]
    result: str
    termination: str
    white_stats: Dict
    black_stats: Dict
    meta: Dict

    # Computed metrics
    material_history: List[float] = field(default_factory=list)
    model_fallbacks: int = 0
    model_decisions: int = 0
    repetition_count: int = 0
    conversion_opportunities: List[Dict] = field(default_factory=list)


def _resolve_game_files(spec: str) -> List[Path]:
    """Resolve one --games argument: file, directory, or glob pattern."""
    p = Path(spec)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    if any(ch in spec for ch in "*?["):
        matches = sorted(p2 for p2 in p.parent.glob(p.name) if p2.is_file())
        if matches:
            return matches
        return sorted(Path.cwd().glob(spec))
    return []


class GameAnalyzer:
    """Analyzes game records to compute various metrics."""

    def analyze_game(self, game_record: Dict) -> GameAnalysis:
        """Analyze a single game record.

        Args:
            game_record: Dictionary with game data

        Returns:
            GameAnalysis object with computed metrics
        """
        analysis = GameAnalysis(
            game_id=str(game_record.get("opening_id", game_record.get("meta", {}).get("seed", "unknown"))),
            moves=game_record.get("moves", []),
            result=game_record.get("result", "*"),
            termination=game_record.get("termination", "unknown"),
            white_stats=game_record.get("white_stats", {}),
            black_stats=game_record.get("black_stats", {}),
            meta=game_record.get("meta", {}),
        )

        model_color = chess.WHITE if analysis.meta.get("model_color", "white") == "white" else chess.BLACK

        # Fallback stats for the model's side
        model_stats = analysis.white_stats if model_color == chess.WHITE else analysis.black_stats
        analysis.model_fallbacks = model_stats.get("fallbacks", 0)
        analysis.model_decisions = model_stats.get("decisions", 0)

        # Replay the game once; track the model's material advantage (POV diff)
        board = chess.Board()
        material_history = []
        position_counts = defaultdict(int)
        position_counts[board.fen()] += 1

        for uci in analysis.moves:
            try:
                board.push_uci(uci)
            except ValueError:
                break
            # Model-POV material advantage after each ply
            material_history.append(material_diff(board, model_color))
            position_counts[board.fen()] += 1

        analysis.material_history = material_history

        # C11: most frequent position across the whole game
        analysis.repetition_count = max(position_counts.values()) if position_counts else 0

        # C8: first crossing of each material-advantage threshold (model POV)
        recorded = set()
        for ply, diff in enumerate(material_history):
            for threshold in CONVERSION_THRESHOLDS:
                if threshold in recorded:
                    continue
                if diff >= threshold:
                    recorded.add(threshold)
                    analysis.conversion_opportunities.append({
                        "ply": ply,
                        "threshold": threshold,
                        "advantage": diff,
                        "result": analysis.result,
                    })

        return analysis

    def analyze_all_games(self, games_spec: List[str]) -> Dict[str, Any]:
        """Analyze all games from files, directories, or glob patterns.

        Args:
            games_spec: List of file paths, directories, or glob patterns

        Returns:
            Dictionary with analysis results
        """
        game_files: List[Path] = []
        seen = set()
        for spec in games_spec:
            for gf in _resolve_game_files(spec):
                if gf not in seen:
                    seen.add(gf)
                    game_files.append(gf)

        all_games = []
        for gf in game_files:
            games = jsonl_read(gf)
            all_games.extend(games)

        print(f"Analyzing {len(all_games)} games from {len(game_files)} files...")

        # Analyze each game
        analyses = []
        for i, game in enumerate(all_games):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{len(all_games)} games...")
            try:
                analyses.append(self.analyze_game(game))
            except Exception as e:
                print(f"  Skipping game {i}: {e}")

        return self._aggregate_results(analyses)

    def _aggregate_results(self, analyses: List[GameAnalysis]) -> Dict[str, Any]:
        """Aggregate analysis results."""

        # C4: Force brute by level
        level_stats = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "total": 0})

        # C8: Conversion rates by threshold
        conversion_stats = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "total": 0})

        # C11: Repetition stats
        repetition_counts = []

        # C12: Score by opening
        opening_stats = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "total": 0, "score": 0})

        for analysis in analyses:
            model_color = analysis.meta.get("model_color", "white")

            # C4: By level
            level_key = str(analysis.meta.get("level", 0))

            if analysis.result == "1-0":
                if model_color == "white":
                    level_stats[level_key]["wins"] += 1
                else:
                    level_stats[level_key]["losses"] += 1
            elif analysis.result == "0-1":
                if model_color == "white":
                    level_stats[level_key]["losses"] += 1
                else:
                    level_stats[level_key]["wins"] += 1
            elif analysis.result == "1/2-1/2":
                level_stats[level_key]["draws"] += 1

            level_stats[level_key]["total"] += 1

            # C8: Conversion (one entry per threshold per game)
            for opp in analysis.conversion_opportunities:
                threshold = str(opp["threshold"])
                if opp["result"] == "1-0":
                    if model_color == "white":
                        conversion_stats[threshold]["wins"] += 1
                    else:
                        conversion_stats[threshold]["losses"] += 1
                elif opp["result"] == "0-1":
                    if model_color == "white":
                        conversion_stats[threshold]["losses"] += 1
                    else:
                        conversion_stats[threshold]["wins"] += 1
                else:
                    conversion_stats[threshold]["draws"] += 1
                conversion_stats[threshold]["total"] += 1

            # C11: Repetitions
            repetition_counts.append(analysis.repetition_count)

            # C12: By opening
            opening_id = analysis.game_id
            if opening_id != "unknown":
                if analysis.result == "1-0":
                    if model_color == "white":
                        opening_stats[opening_id]["wins"] += 1
                    else:
                        opening_stats[opening_id]["losses"] += 1
                elif analysis.result == "0-1":
                    if model_color == "white":
                        opening_stats[opening_id]["losses"] += 1
                    else:
                        opening_stats[opening_id]["wins"] += 1
                elif analysis.result == "1/2-1/2":
                    opening_stats[opening_id]["draws"] += 1
                opening_stats[opening_id]["total"] += 1

        # Compute scores
        for level, stats in level_stats.items():
            total = stats["total"]
            if total > 0:
                stats["score"] = (stats["wins"] + 0.5 * stats["draws"]) / total

        for threshold, stats in conversion_stats.items():
            total = stats["total"]
            if total > 0:
                stats["conversion_rate"] = stats["wins"] / total
                stats["draw_rate"] = stats["draws"] / total
                stats["loss_rate"] = stats["losses"] / total

        # C12: Bootstrap confidence intervals
        opening_scores = {}
        for opening_id, stats in opening_stats.items():
            total = stats["total"]
            if total > 0:
                opening_scores[opening_id] = (stats["wins"] + 0.5 * stats["draws"]) / total

        # Compute bootstrap CI (seeded for determinism)
        if len(opening_scores) > 10:
            scores_list = list(opening_scores.values())
            n_bootstrap = 10000
            rng = random.Random(0)
            bootstrap_means = []

            for _ in range(n_bootstrap):
                sample = rng.choices(scores_list, k=len(scores_list))
                bootstrap_means.append(sum(sample) / len(sample))

            bootstrap_means.sort()
            lower = bootstrap_means[int(0.025 * n_bootstrap)]
            upper = bootstrap_means[int(0.975 * n_bootstrap)]
            mean_score = sum(scores_list) / len(scores_list)
        else:
            lower = upper = mean_score = sum(opening_scores.values()) / len(opening_scores) if opening_scores else 0

        # Sort openings by score
        sorted_openings = sorted(opening_scores.items(), key=lambda x: x[1])

        return {
            "C4_by_level": dict(level_stats),
            "C8_conversion": dict(conversion_stats),
            "C11_repetitions": {
                "max": max(repetition_counts) if repetition_counts else 0,
                "mean": sum(repetition_counts) / len(repetition_counts) if repetition_counts else 0,
                "count_ge3": sum(1 for c in repetition_counts if c >= 3),
            },
            "C12_variance": {
                "mean_score": mean_score,
                "ci_95": [lower, upper],
                "best_openings": sorted_openings[-5:] if len(sorted_openings) >= 5 else sorted_openings,
                "worst_openings": sorted_openings[:5] if len(sorted_openings) >= 5 else sorted_openings,
                "n_openings": len(opening_scores),
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Analyze game records")
    parser.add_argument("--games", type=str, nargs="+", required=True,
                        help="Game files, directories, or glob patterns")
    parser.add_argument("--out", type=str, default="results/analysis/analysis_base.json",
                        help="Output file path (a directory writes analysis_base.json inside)")
    args = parser.parse_args()

    # Create analyzer
    analyzer = GameAnalyzer()

    # Analyze games
    results = analyzer.analyze_all_games(args.games)

    # Save results
    out_path = Path(args.out)
    if out_path.is_dir() or out_path.suffix != ".json":
        out_path = out_path / "analysis_base.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Analysis saved to {out_path}")


if __name__ == "__main__":
    main()
