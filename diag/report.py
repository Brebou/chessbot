"""Generate capability profile report from evaluation results.

Aggregates results from:
- Evaluation vs engine (C4)
- Game analysis (C8 fallback, C11, C12)
- Probes (C1-C3, C5-C7 from P1-P4; C8 from P5; C9 from P6; C10 from P6b;
  C13 from P7; C14 from P1)

Produces:
- profile.md (human-readable)
- profile.json (machine-readable)

Usage:
    uv run diag/report.py --eval results/eval_base --analysis results/analysis \
                          --probes results/probes --out results/profile

All three input arguments accept either a JSON file or a directory (the
canonical file inside the directory is picked automatically, e.g.
``summary.json`` for --eval, ``analysis_base.json`` for --analysis and
``probes_base.json`` for --probes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _weighted_avg(entries: List[dict], key: str) -> float:
    """Weighted average of ``key`` over entries that have it, by weight ``n``."""
    num = sum(e[key] * e["n"] for e in entries if key in e and e.get("n"))
    den = sum(e["n"] for e in entries if key in e and e.get("n"))
    return num / den if den else 0.0


class ReportGenerator:
    """Generates capability profile report."""

    def __init__(self):
        pass

    def generate_report(self, eval_path: Path, analysis_path: Path, probes_path: Path,
                        out_dir: Path) -> dict:
        """Generate comprehensive capability profile.

        Args:
            eval_path: Path to evaluation summary JSON (or its directory)
            analysis_path: Path to game analysis JSON (or its directory)
            probes_path: Path to probes results JSON (or its directory)
            out_dir: Output directory

        Returns:
            Dictionary with report data
        """
        # Load all data
        eval_data = self._load_json(eval_path, ["summary.json"])
        analysis_data = self._load_json(analysis_path, ["analysis_base.json", "analysis.json"])
        probes_data = self._load_json(probes_path, ["probes_base.json", "probes_results.json"])

        eval_data = self._aggregate_eval(eval_data)

        # Build report
        capabilities = self._build_capabilities(eval_data, analysis_data, probes_data)
        self._last_capabilities = capabilities
        report = {
            "model": eval_data.get("model", "unknown"),
            "capabilities": capabilities,
            "top_weaknesses": self._identify_weaknesses(eval_data, analysis_data, probes_data),
            "recommendations": self._generate_recommendations(eval_data, analysis_data, probes_data),
        }

        # Generate markdown
        md_content = self._generate_markdown(report)

        # Save files
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = out_dir / "profile.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Save markdown
        md_path = out_dir / "profile.md"
        with open(md_path, 'w') as f:
            f.write(md_content)

        print(f"✅ Report generated:")
        print(f"   JSON: {json_path}")
        print(f"   Markdown: {md_path}")

        return report

    def _load_json(self, path: Path, dir_roles: List[str]) -> Dict[str, Any]:
        """Load a JSON file; when ``path`` is a directory, try ``dir_roles``."""
        path = Path(path)
        if path.is_dir():
            for name in dir_roles:
                candidate = path / name
                if candidate.exists():
                    path = candidate
                    break
            else:
                print(f"⚠️ No known result file found in {path} (tried {dir_roles})")
                return {}
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"⚠️ Could not load {path}: {e}")
            return {}

    def _aggregate_eval(self, eval_data: Dict) -> Dict[str, Any]:
        """Aggregate multi-seed eval summaries ({seed_X: summary}) into one dict.

        Flat summaries (with top-level per_level/mean_score) pass through.
        """
        if "per_level" in eval_data:
            return eval_data

        seeds = {k: v for k, v in eval_data.items()
                 if isinstance(v, dict) and "per_level" in v}
        if not seeds:
            return {}

        levels = sorted({lvl for v in seeds.values() for lvl in v["per_level"]},
                        key=lambda x: int(x))
        per_level = {}
        for lvl in levels:
            entries = [v["per_level"][lvl] for v in seeds.values()
                       if lvl in v.get("per_level", {})]
            per_level[lvl] = {
                "games": sum(e.get("games", 0) for e in entries),
                "score": sum(e.get("score", 0.0) for e in entries) / len(entries),
            }

        aggregated = dict(next(iter(seeds.values())))
        aggregated["per_level"] = per_level
        aggregated["mean_score"] = sum(v.get("mean_score", 0.0) for v in seeds.values()) / len(seeds)
        aggregated["n_seeds"] = len(seeds)
        return aggregated

    def _build_capabilities(self, eval_data: Dict, analysis_data: Dict, probes_data: Dict) -> Dict:

        capabilities = {}

        # C1-C3, C14: From P1 probe
        if "P1" in probes_data:
            p1 = probes_data["P1"]
            total = p1.get("metrics", {}).get("total", {})

            if "top1_legal_pct" in total:
                capabilities["C1"] = {
                    "description": "Légalité conditionnelle (top-1/top-3/fallback)",
                    "top1_legal_pct": total["top1_legal_pct"],
                    "top3_legal_pct": total.get("top3_legal_pct", 0),
                    "fallback_pct": total.get("fallback_pct", 0),
                }

                capabilities["C2"] = {
                    "description": "Taxonomie des illégaux",
                    "by_bucket": p1.get("by_bucket", {}),
                }

                capabilities["C3"] = {
                    "description": "Masse sur les coups légaux",
                    "mass_legal_avg": total.get("mass_legal_avg", 0),
                    "mass_top1_avg": total.get("mass_top1_avg", 0),
                    "mass_top3_avg": total.get("mass_top3_avg", 0),
                }

            # C14: Stability at end of context (endgame vs opening buckets)
            buckets = p1.get("by_bucket", {})
            opening = buckets.get("opening", {})
            endgame = buckets.get("endgame", {})
            if opening and endgame:
                capabilities["C14"] = {
                    "description": "Stabilité fin de contexte",
                    "opening_top1_legal_pct": opening.get("top1_legal_pct", 0),
                    "endgame_top1_legal_pct": endgame.get("top1_legal_pct", 0),
                    "top1_legal_delta": endgame.get("top1_legal_pct", 0) - opening.get("top1_legal_pct", 0),
                    "opening_mass_legal_avg": opening.get("mass_legal_avg", 0),
                    "endgame_mass_legal_avg": endgame.get("mass_legal_avg", 0),
                    "mass_legal_delta": endgame.get("mass_legal_avg", 0) - opening.get("mass_legal_avg", 0),
                }

        # C4: From evaluation
        if "per_level" in eval_data:
            capabilities["C4"] = {
                "description": "Force brute (score vs moteur)",
                "mean_score": eval_data.get("mean_score", 0),
                "by_level": eval_data.get("per_level", {}),
                "n_seeds": eval_data.get("n_seeds", 1),
            }

        # C5: From P2 probe
        if "P2" in probes_data:
            p2 = probes_data["P2"]
            m = p2.get("metrics", {})
            if "top1_mate_pct" in m:
                capabilities["C5"] = {
                    "description": "Mat en 1",
                    "top1_mate_pct": m["top1_mate_pct"],
                    "top3_mate_pct": m.get("top3_mate_pct", 0),
                    "check_not_mate_count": m.get("check_not_mate_count", 0),
                }

        # C6: From P3 probe
        if "P3" in probes_data:
            p3 = probes_data["P3"]
            m = p3.get("metrics", {})
            if "take_winning_capture_pct" in m:
                capabilities["C6"] = {
                    "description": "Captures",
                    "take_winning_pct": m["take_winning_capture_pct"],
                    "take_losing_pct": m.get("take_losing_capture_pct", 0),
                    "take_free_pct": m.get("take_free_piece_pct", 0),
                }

        # C7: From P4 probe
        if "P4" in probes_data:
            p4 = probes_data["P4"]
            m = p4.get("metrics", {})
            if "hang_1pt_pct" in m:
                capabilities["C7"] = {
                    "description": "Anti-hang (perte matériel)",
                    "hang_1pt_pct": m["hang_1pt_pct"],
                    "hang_3pt_pct": m.get("hang_3pt_pct", 0),
                }

        # C8: From P5 probe (preferred), else from game analysis
        c8_from_p5 = None
        if "P5" in probes_data:
            p5 = probes_data["P5"]
            metrics = p5.get("metrics", {})
            thresholds = sorted(
                {thr for lvl in metrics.values() if isinstance(lvl, dict)
                 for thr in lvl if str(thr).isdigit()},
                key=int)
            by_threshold = {}
            for thr in thresholds:
                entries = [lvl[thr] for lvl in metrics.values()
                           if isinstance(lvl, dict) and thr in lvl]
                n = sum(e.get("n", 0) for e in entries)
                if n:
                    win_rate = _weighted_avg(entries, "win_rate")
                    by_threshold[thr] = {
                        "n": n,
                        "win_rate": win_rate,
                        "conversion_rate": win_rate,
                        "draw_rate": _weighted_avg(entries, "draw_rate"),
                        "loss_rate": _weighted_avg(entries, "loss_rate"),
                    }
            if by_threshold:
                c8_from_p5 = by_threshold

        if c8_from_p5:
            capabilities["C8"] = {
                "description": "Conversion de l'avantage",
                "by_threshold": c8_from_p5,
                "source": "P5",
            }
        elif "C8_conversion" in analysis_data and analysis_data["C8_conversion"]:
            capabilities["C8"] = {
                "description": "Conversion de l'avantage",
                "by_threshold": analysis_data["C8_conversion"],
                "source": "game_analysis",
            }

        # C9: From P6 probe
        if "P6" in probes_data:
            p6 = probes_data["P6"]
            metrics = p6.get("metrics", {})
            entries = [stats for types in metrics.values() if isinstance(types, dict)
                       for stats in types.values()
                       if isinstance(stats, dict) and "mate_rate" in stats]
            if entries:
                mate_plies = [e["avg_plies_to_mate"] for e in entries
                              if e.get("avg_plies_to_mate") is not None]
                failure_modes: Dict[str, int] = {}
                for e in entries:
                    for cause, count in e.get("failure_modes", {}).items():
                        failure_modes[cause] = failure_modes.get(cause, 0) + count
                capabilities["C9"] = {
                    "description": "Fins de parties (KQK/KRK)",
                    "n": sum(e.get("n", 0) for e in entries),
                    "mate_rate": _weighted_avg(entries, "mate_rate"),
                    "mate_le50_rate": _weighted_avg(entries, "mate_le50_rate"),
                    "mate_le100_rate": _weighted_avg(entries, "mate_le100_rate"),
                    "avg_plies_to_mate": (sum(mate_plies) / len(mate_plies)) if mate_plies else None,
                    "failure_modes": failure_modes,
                    "by_type": p6.get("by_bucket", {}),
                }

        # C10: From P6b probe
        if "P6b" in probes_data:
            m = probes_data["P6b"].get("metrics", {})
            if "queen_rate_when_safe" in m:
                capabilities["C10"] = {
                    "description": "Promotion",
                    "n_with_safe_promotion": m.get("n_with_safe_promotion", 0),
                    "queen_rate_when_safe": m["queen_rate_when_safe"],
                    "promote_rate_when_safe": m.get("promote_rate_when_safe", 0),
                    "promote_rate_overall": m.get("promote_rate_overall", 0),
                    "piece_distribution_pct": m.get("piece_distribution_pct", {}),
                }

        # C11: From analysis
        if "C11_repetitions" in analysis_data:
            c11 = analysis_data["C11_repetitions"]
            capabilities["C11"] = {
                "description": "Shuffling/répétition",
                "max_repetitions": c11.get("max", 0),
                "mean_repetitions": c11.get("mean", 0),
                "count_ge3": c11.get("count_ge3", 0),
            }

        # C12: From analysis
        if "C12_variance" in analysis_data:
            c12 = analysis_data["C12_variance"]
            capabilities["C12"] = {
                "description": "Variance & robustesse",
                "mean_score": c12.get("mean_score", 0),
                "ci_95": c12.get("ci_95", [0, 0]),
                "best_openings": c12.get("best_openings", []),
                "worst_openings": c12.get("worst_openings", []),
            }

        # C13: From P7 probe
        if "P7" in probes_data:
            m = probes_data["P7"].get("metrics", {})
            if m.get("status") == "skipped":
                capabilities["C13"] = {
                    "description": "Généralité next-move",
                    "status": "SKIPPED",
                    "reason": m.get("reason", "dataset unavailable"),
                }
            elif "unseen" in m:
                seen = m.get("seen", {})
                unseen = m.get("unseen", {})
                capabilities["C13"] = {
                    "description": "Généralité next-move",
                    "split_index": m.get("split_index"),
                    "seen_top1_acc": seen.get("top1_acc", 0),
                    "seen_top3_acc": seen.get("top3_acc", 0),
                    "seen_n": seen.get("n", 0),
                    "unseen_top1_acc": unseen.get("top1_acc", 0),
                    "unseen_top3_acc": unseen.get("top3_acc", 0),
                    "unseen_n": unseen.get("n", 0),
                    "elo_tranches_unseen": m.get("elo_tranches_unseen", {}),
                }

        return capabilities

    def _identify_weaknesses(self, eval_data: Dict, analysis_data: Dict, probes_data: Dict) -> list:

        weaknesses = []

        # C4: Low score vs engine
        if "mean_score" in eval_data:
            score = eval_data["mean_score"]
            if score < 0.3:
                weaknesses.append({
                    "id": "C4_low_score",
                    "description": f"Score moyen faible vs moteur: {score:.3f}",
                    "impact": "Énorme - le modèle perd contre un moteur simple",
                    "estimated_elo_loss": "~200-300 Elo",
                })
            elif score < 0.5:
                weaknesses.append({
                    "id": "C4_medium_score",
                    "description": f"Score moyen moyen vs moteur: {score:.3f}",
                    "impact": "Significatif - amélioration possible",
                    "estimated_elo_loss": "~100-200 Elo",
                })

        # C1: High fallback rate
        fallback = probes_data.get("P1", {}).get("metrics", {}).get("total", {}).get("fallback_pct", 0)
        if fallback > 5:
            weaknesses.append({
                "id": "C1_high_fallback",
                "description": f"Taux de fallback élevé: {fallback:.1f}%",
                "impact": "Le modèle propose souvent des coups illégaux",
                "estimated_elo_loss": f"~{fallback*2} Elo",
            })

        # C5: Low mate-in-1 detection
        top1_mate = probes_data.get("P2", {}).get("metrics", {}).get("top1_mate_pct", 0)
        if probes_data.get("P2", {}).get("n_positions", 0) > 0 and top1_mate < 50:
            weaknesses.append({
                "id": "C5_low_mate_detection",
                "description": f"Détection mat en 1 faible: {top1_mate:.1f}%",
                "impact": "Le modèle ne voit pas les mats simples",
                "estimated_elo_loss": "~50-100 Elo",
            })

        # C6: Poor capture decisions
        p3_metrics = probes_data.get("P3", {}).get("metrics", {})
        if p3_metrics.get("n", 0) > 0:
            take_winning = p3_metrics.get("take_winning_capture_pct", 0)
            take_losing = p3_metrics.get("take_losing_capture_pct", 0)

            if take_winning < 70:
                weaknesses.append({
                    "id": "C6_poor_capture_decision",
                    "description": f"Prend les captures gagnantes seulement {take_winning:.1f}% du temps",
                    "impact": "Perte de matériel évitable",
                    "estimated_elo_loss": "~30-50 Elo",
                })

            if take_losing > 10:
                weaknesses.append({
                    "id": "C6_takes_losing_captures",
                    "description": f"Prend des captures perdantes {take_losing:.1f}% du temps",
                    "impact": "Perte de matériel inutile",
                    "estimated_elo_loss": "~20-30 Elo",
                })

        # C7: High hang rate
        hang_3pt = probes_data.get("P4", {}).get("metrics", {}).get("hang_3pt_pct", 0)
        if hang_3pt > 5:
            weaknesses.append({
                "id": "C7_high_hang_rate",
                "description": f"Taux de hang ≥3pts: {hang_3pt:.1f}%",
                "impact": "Le modèle abandonne du matériel gratuitement",
                "estimated_elo_loss": f"~{hang_3pt*3} Elo",
            })

        # C8: Poor conversion (prefer probe data, fall back to analysis)
        c8 = None
        if "C8" in (capabilities_cache := getattr(self, "_last_capabilities", {})):
            c8 = capabilities_cache["C8"].get("by_threshold")
        if c8 is None:
            c8 = analysis_data.get("C8_conversion", {})
        for threshold, stats in c8.items():
            conversion = stats.get("conversion_rate", 0)
            if conversion < 0.7:
                weaknesses.append({
                    "id": f"C8_poor_conversion_{threshold}",
                    "description": f"Conversion faible depuis +{threshold}: {conversion*100:.1f}%",
                    "impact": "Ne sait pas convertir l'avantage",
                    "estimated_elo_loss": "~20-40 Elo",
                })

        # C9: Poor endgame technique
        p6_metrics = probes_data.get("P6", {}).get("metrics", {})
        mate_le50 = [
            stats.get("mate_le50_rate", 0)
            for types in p6_metrics.values() if isinstance(types, dict)
            for stats in types.values() if isinstance(stats, dict) and stats.get("n", 0) > 0
        ]
        if mate_le50 and sum(mate_le50) / len(mate_le50) < 0.3:
            rate = sum(mate_le50) / len(mate_le50)
            weaknesses.append({
                "id": "C9_weak_endgames",
                "description": f"Fins de parties: mate en ≤50 coups seulement {rate*100:.1f}%",
                "impact": "Ne convertit pas les gains techniques KQK/KRK",
                "estimated_elo_loss": "~30-60 Elo",
            })

        # C13: Overfitting (unseen much worse than seen)
        p7_metrics = probes_data.get("P7", {}).get("metrics", {})
        if p7_metrics.get("unseen", {}).get("n", 0) > 0 and p7_metrics.get("seen", {}).get("n", 0) > 0:
            seen_acc = p7_metrics["seen"].get("top1_acc", 0)
            unseen_acc = p7_metrics["unseen"].get("top1_acc", 0)
            if unseen_acc < seen_acc - 0.10:
                weaknesses.append({
                    "id": "C13_overfitting",
                    "description": f"Précision next-move en baisse sur données vues/inédites: "
                                   f"{seen_acc*100:.1f}% vs {unseen_acc*100:.1f}%",
                    "impact": "Mémorisation plutôt que généralisation",
                    "estimated_elo_loss": "~20-50 Elo",
                })

        # Sort by estimated Elo loss
        def _elo(w):
            try:
                return float(w["estimated_elo_loss"].replace("~", "").replace(" Elo", "").split("-")[0])
            except (ValueError, AttributeError):
                return 0.0

        weaknesses.sort(key=_elo, reverse=True)

        return weaknesses[:5]  # Top 5

    def _generate_recommendations(self, eval_data: Dict, analysis_data: Dict, probes_data: Dict) -> list:

        recommendations = []

        # Based on C4 (score vs engine)
        if "mean_score" in eval_data:
            score = eval_data["mean_score"]
            if score < 0.5:
                recommendations.append({
                    "priority": "HIGH",
                    "action": "Fine-tune on engine games (levels 0-3)",
                    "objective": f"Improve score from {score:.3f} to >0.6",
                    "method": "SFT with engine-labeled moves",
                    "estimated_gain": "+100-200 Elo",
                })

        # Based on C1 (legality)
        fallback = probes_data.get("P1", {}).get("metrics", {}).get("total", {}).get("fallback_pct", 0)
        if fallback > 3:
            recommendations.append({
                "priority": "HIGH",
                "action": "Legal finetuning with legality loss",
                "objective": f"Reduce fallback rate from {fallback:.1f}% to <2%",
                "method": "Train on on-policy positions with legality mask",
                "estimated_gain": "+50-100 Elo",
            })

        # Based on C5 (mate in 1)
        p2_metrics = probes_data.get("P2", {}).get("metrics", {})
        if probes_data.get("P2", {}).get("n_positions", 0) > 0:
            top1_mate = p2_metrics.get("top1_mate_pct", 0)
            if top1_mate < 70:
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": "Mate-in-1 finetuning",
                    "objective": f"Increase mate-in-1 detection from {top1_mate:.1f}% to >80%",
                    "method": "SFT on mate-in-1 positions with high weight",
                    "estimated_gain": "+30-50 Elo",
                })

        # Based on C6 (captures)
        p3_metrics = probes_data.get("P3", {}).get("metrics", {})
        if p3_metrics.get("n", 0) > 0:
            take_winning = p3_metrics.get("take_winning_capture_pct", 0)
            if take_winning < 80:
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": "Capture finetuning with SEE labels",
                    "objective": f"Increase winning capture rate from {take_winning:.1f}% to >85%",
                    "method": "SFT with SEE-weighted capture loss",
                    "estimated_gain": "+20-40 Elo",
                })

        # Based on C7 (hanging)
        hang_3pt = probes_data.get("P4", {}).get("metrics", {}).get("hang_3pt_pct", 0)
        if hang_3pt > 3:
            recommendations.append({
                "priority": "MEDIUM",
                "action": "Anti-hang finetuning",
                "objective": f"Reduce hang rate from {hang_3pt:.1f}% to <2%",
                "method": "RL with material loss penalty",
                "estimated_gain": "+20-30 Elo",
            })

        # Based on C8 (conversion, prefer probe data)
        c8 = analysis_data.get("C8_conversion", {})
        for threshold, stats in c8.items():
            conversion = stats.get("conversion_rate", 0)
            if conversion < 0.75:
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": f"Conversion training from +{threshold}",
                    "objective": f"Increase conversion from {conversion*100:.1f}% to >80%",
                    "method": "RL with win/draw reward shaping",
                    "estimated_gain": "+15-25 Elo",
                })
                break  # Only one recommendation for C8

        # Based on C9 (endgames)
        p6_metrics = probes_data.get("P6", {}).get("metrics", {})
        mate_rates = [
            stats.get("mate_le50_rate", 0)
            for types in p6_metrics.values() if isinstance(types, dict)
            for stats in types.values() if isinstance(stats, dict) and stats.get("n", 0) > 0
        ]
        if mate_rates and sum(mate_rates) / len(mate_rates) < 0.5:
            recommendations.append({
                "priority": "MEDIUM",
                "action": "Endgame technique training (KQK/KRK)",
                "objective": "Increase mate-in-50-plies rate to >60%",
                "method": "SFT on curated endgame histories with engine annotations",
                "estimated_gain": "+20-40 Elo",
            })

        # Based on C13 (generality)
        p7_metrics = probes_data.get("P7", {}).get("metrics", {})
        if p7_metrics.get("unseen", {}).get("n", 0) > 0 and p7_metrics.get("seen", {}).get("n", 0) > 0:
            seen_acc = p7_metrics["seen"].get("top1_acc", 0)
            unseen_acc = p7_metrics["unseen"].get("top1_acc", 0)
            if unseen_acc < seen_acc - 0.10:
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": "Data augmentation for generality",
                    "objective": f"Close the seen/unseen accuracy gap ({seen_acc*100:.1f}% vs {unseen_acc*100:.1f}%)",
                    "method": "Diversify training games (more players, more openings)",
                    "estimated_gain": "+10-30 Elo",
                })

        # Sort by priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 2))

        return recommendations

    def _generate_markdown(self, report: Dict) -> str:
        """Generate markdown report."""

        md = []
        md.append("# ChessGPT Model Capability Profile")
        md.append("")
        md.append(f"**Model:** {report['model']}")
        md.append(f"**Date:** {report.get('date', 'N/A')}")
        md.append("")

        # Capabilities table
        md.append("## 📊 Capabilities (C1-C14)")
        md.append("")
        md.append("| # | Capability | Metric | Value |")
        md.append("|---|------------|--------|-------|")

        for cid, cap in report["capabilities"].items():
            if isinstance(cap, dict) and "description" in cap:
                # Simplify for table
                if "top1_legal_pct" in cap:
                    val = f"{cap['top1_legal_pct']:.1f}% / {cap['top3_legal_pct']:.1f}%"
                elif "mass_legal_avg" in cap and "top1_legal_pct" not in cap:
                    val = f"{cap['mass_legal_avg']:.3f}"
                elif "mean_score" in cap and "ci_95" not in cap:
                    val = f"{cap['mean_score']:.3f}"
                elif "top1_mate_pct" in cap:
                    val = f"top-1 {cap['top1_mate_pct']:.1f}% / top-3 {cap['top3_mate_pct']:.1f}%"
                elif "take_winning_pct" in cap:
                    val = (f"win {cap['take_winning_pct']:.1f}% / "
                           f"lose {cap['take_losing_pct']:.1f}% / free {cap['take_free_pct']:.1f}%")
                elif "hang_1pt_pct" in cap:
                    val = f"≥1pt {cap['hang_1pt_pct']:.1f}% / ≥3pts {cap['hang_3pt_pct']:.1f}%"
                elif "mate_rate" in cap:
                    val = (f"mate {cap['mate_rate']*100:.1f}% / ≤50 {cap['mate_le50_rate']*100:.1f}% / "
                           f"≤100 {cap['mate_le100_rate']*100:.1f}%")
                elif "queen_rate_when_safe" in cap:
                    val = f"queen|safe {cap['queen_rate_when_safe']*100:.1f}%"
                elif "unseen_top1_acc" in cap:
                    val = (f"seen {cap['seen_top1_acc']*100:.1f}% / "
                           f"unseen {cap['unseen_top1_acc']*100:.1f}% (top-1)")
                elif "top1_legal_delta" in cap:
                    val = f"Δ top-1 legal (endgame-opening): {cap['top1_legal_delta']:.1f} pts"
                elif "by_threshold" in cap:
                    val = ", ".join(
                        f"+{k}: {v.get('conversion_rate', v.get('win_rate', 0))*100:.1f}%"
                        for k, v in cap["by_threshold"].items())
                elif "by_bucket" in cap:
                    phases = ["opening", "early_mid", "mid", "late_mid", "endgame"]
                    val = ", ".join(
                        f"{p}: {cap['by_bucket'][p].get('fallback_pct', 0):.1f}%"
                        for p in phases if p in cap["by_bucket"])
                elif "max_repetitions" in cap:
                    val = f"max {cap['max_repetitions']} / mean {cap['mean_repetitions']:.2f} / ≥3: {cap['count_ge3']}"
                elif "ci_95" in cap:
                    val = f"{cap['mean_score']:.3f} (CI95 {cap['ci_95'][0]:.3f}-{cap['ci_95'][1]:.3f})"
                elif "status" in cap:
                    val = f"{cap['status']}: {cap.get('reason', '')}"
                else:
                    val = json.dumps(cap, indent=0)[:50]

                md.append(f"| {cid} | {cap['description']} | - | {val} |")

        md.append("")

        # Top weaknesses
        md.append("## 🎯 Top 5 Weaknesses")
        md.append("")

        if not report["top_weaknesses"]:
            md.append("_No significant weakness detected on the available data._")
            md.append("")

        for i, weakness in enumerate(report["top_weaknesses"], 1):
            md.append(f"### {i}. {weakness['description']}")
            md.append(f"- **Impact:** {weakness['impact']}")
            md.append(f"- **Estimated Elo loss:** {weakness['estimated_elo_loss']}")
            md.append("")

        # Recommendations
        md.append("## 🚀 Training Recommendations")
        md.append("")

        if not report["recommendations"]:
            md.append("_No training recommendation triggered by the available data._")
            md.append("")

        for i, rec in enumerate(report["recommendations"], 1):
            md.append(f"### {i}. [{rec['priority']}] {rec['action']}")
            md.append(f"- **Objective:** {rec['objective']}")
            md.append(f"- **Method:** {rec['method']}")
            md.append(f"- **Estimated gain:** {rec['estimated_gain']}")
            md.append("")

        return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Generate capability profile report")
    parser.add_argument("--eval", type=str, required=True,
                        help="Path to evaluation summary JSON (or its directory)")
    parser.add_argument("--analysis", type=str, required=True,
                        help="Path to game analysis JSON (or its directory)")
    parser.add_argument("--probes", type=str, required=True,
                        help="Path to probes results JSON (or its directory)")
    parser.add_argument("--out", type=str, default="results/profile",
                        help="Output directory")
    args = parser.parse_args()

    # Create generator
    generator = ReportGenerator()

    # Generate report
    generator.generate_report(
        eval_path=Path(args.eval),
        analysis_path=Path(args.analysis),
        probes_path=Path(args.probes),
        out_dir=Path(args.out)
    )


if __name__ == "__main__":
    main()
