"""Analyze games to extract metrics for C4, C8, C11, C12.

This module analyzes game records to compute:
- C4: Force brute (score vs engine by level)
- C8: Conversion de l'avantage (win rate from +2/+3/+5/+9)
- C11: Shuffling/repetition analysis
- C12: Variance and robustness

Usage:
    uv run diag/analyze_games.py --games results/eval_base/*.jsonl --out results/analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import random

import chess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .common import material_value, material_diff, jsonl_read, jsonl_write


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
    fallbacks_by_ply: Dict[int, int] = field(default_factory=dict)
    repetition_count: int = 0
    conversion_opportunities: List[Dict] = field(default_factory=list)


class GameAnalyzer:
    """Analyzes game records to compute various metrics."""
    
    def __init__(self):
        pass
    
    def analyze_game(self, game_record: Dict) -> GameAnalysis:
        """Analyze a single game record.
        
        Args:
            game_record: Dictionary with game data
            
        Returns:
            GameAnalysis object with computed metrics
        """
        analysis = GameAnalysis(
            game_id=game_record.get("opening_id", "unknown"),
            moves=game_record.get("moves", []),
            result=game_record.get("result", "*"),
            termination=game_record.get("termination", "unknown"),
            white_stats=game_record.get("white_stats", {}),
            black_stats=game_record.get("black_stats", {}),
            meta=game_record.get("meta", {}),
        )
        
        # Compute material history
        board = chess.Board()
        material_history = []
        
        for i, uci in enumerate(analysis.moves):
            try:
                board.push_uci(uci)
                mat = material_value(board, chess.WHITE)
                material_history.append(mat)
            except:
                break
        
        analysis.material_history = material_history
        
        # Track fallbacks
        if "white_stats" in game_record:
            for attempt_idx, count in enumerate(game_record["white_stats"].get("attempts", [0, 0, 0])):
                # Fallback = attempt 0 (all top-3 illegal)
                if attempt_idx == 2:  # This is the fallback count
                    for ply in range(len(analysis.moves)):
                        if board.turn == chess.WHITE:
                            analysis.fallbacks_by_ply[ply] = count
        
        # Count repetitions
        position_counts = defaultdict(int)
        for i, uci in enumerate(analysis.moves):
            try:
                if i % 2 == 0:  # After white's move
                    fen = board.fen()
                    position_counts[fen] += 1
            except:
                pass
        
        analysis.repetition_count = max(position_counts.values()) if position_counts else 0
        
        # Find conversion opportunities (+2, +3, +5, +9 material)
        for i, mat in enumerate(material_history):
            if mat >= 2:
                # Check if this is a conversion opportunity
                # Look ahead to see if advantage was maintained
                for j in range(i, min(i + 20, len(material_history))):
                    if material_history[j] < mat - 1:  # Lost advantage
                        break
                else:
                    # Advantage maintained for 20+ plies
                    analysis.conversion_opportunities.append({
                        "ply": i,
                        "advantage": mat,
                        "result": analysis.result,
                    })
        
        return analysis
    
    def analyze_all_games(self, games_dir: Path) -> Dict[str, Any]:
        """Analyze all games in a directory.
        
        Args:
            games_dir: Directory containing game files
            
        Returns:
            Dictionary with analysis results
        """
        # Collect all game files
        game_files = list(Path(games_dir).glob("*.jsonl"))
        
        all_games = []
        for gf in game_files:
            games = jsonl_read(gf)
            all_games.extend(games)
        
        print(f"Analyzing {len(all_games)} games...")
        
        # Analyze each game
        analyses = []
        for i, game in enumerate(all_games):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(all_games)} games...")
            analysis = self.analyze_game(game)
            analyses.append(analysis)
        
        # Compute aggregate metrics
        results = self._aggregate_results(analyses)
        
        return results
    
    def _aggregate_results(self, analyses: List[GameAnalysis]) -> Dict[str, Any]:
        """Aggregate analysis results."""
        
        # C4: Force brute by level
        level_stats = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "total": 0})
        
        # C8: Conversion rates
        conversion_stats = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "total": 0})
        
        # C11: Repetition stats
        repetition_counts = []
        
        # C12: Score by opening
        opening_stats = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "total": 0, "score": 0})
        
        for analysis in analyses:
            # C4: By level
            level = analysis.meta.get("level", 0)
            level_key = str(level)
            
            if analysis.result == "1-0":
                if analysis.meta.get("model_color") == "white":
                    level_stats[level_key]["wins"] += 1
                else:
                    level_stats[level_key]["losses"] += 1
            elif analysis.result == "0-1":
                if analysis.meta.get("model_color") == "white":
                    level_stats[level_key]["losses"] += 1
                else:
                    level_stats[level_key]["wins"] += 1
            elif analysis.result == "1/2-1/2":
                level_stats[level_key]["draws"] += 1
            
            level_stats[level_key]["total"] += 1
            
            # C8: Conversion
            for opp in analysis.conversion_opportunities:
                advantage = opp["advantage"]
                threshold = None
                if advantage >= 9:
                    threshold = ">=9"
                elif advantage >= 5:
                    threshold = ">=5"
                elif advantage >= 3:
                    threshold = ">=3"
                elif advantage >= 2:
                    threshold = ">=2"
                
                if threshold:
                    if opp["result"] == "1-0":
                        conversion_stats[threshold]["wins"] += 1
                    elif opp["result"] == "0-1":
                        conversion_stats[threshold]["losses"] += 1
                    else:
                        conversion_stats[threshold]["draws"] += 1
                    conversion_stats[threshold]["total"] += 1
            
            # C11: Repetitions
            repetition_counts.append(analysis.repetition_count)
            
            # C12: By opening
            opening_id = analysis.game_id
            if opening_id != "unknown":
                if analysis.result == "1-0":
                    if analysis.meta.get("model_color") == "white":
                        opening_stats[opening_id]["wins"] += 1
                    else:
                        opening_stats[opening_id]["losses"] += 1
                elif analysis.result == "0-1":
                    if analysis.meta.get("model_color") == "white":
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
        
        # Compute bootstrap CI
        if len(opening_scores) > 10:
            scores_list = list(opening_scores.values())
            n_bootstrap = 10000
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                sample = random.choices(scores_list, k=len(scores_list))
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
            "C4_by_level": level_stats,
            "C8_conversion": conversion_stats,
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
    parser.add_argument("--games", type=str, required=True,
                        help="Directory or glob pattern for game files")
    parser.add_argument("--out", type=str, default="results/analysis.json",
                        help="Output file path")
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = GameAnalyzer()
    
    # Analyze games
    results = analyzer.analyze_all_games(Path(args.games))
    
    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Analysis saved to {out_path}")


if __name__ == "__main__":
    main()
