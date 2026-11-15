"""Generate capability profile report from evaluation results.

Aggregates results from:
- Evaluation vs engine (C4)
- Game analysis (C8, C11, C12)
- Probes (C1-C3, C5-C7, C9-C14)

Produces:
- profile.md (human-readable)
- profile.json (machine-readable)

Usage:
    uv run diag/report.py --eval results/eval_base --analysis results/analysis.json \
                          --probes results/probes.json --out results/profile
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


class ReportGenerator:
    """Generates capability profile report."""
    
    def __init__(self):
        pass
    
    def generate_report(self, eval_path: Path, analysis_path: Path, probes_path: Path,
                        out_dir: Path) -> dict:
        """Generate comprehensive capability profile.
        
        Args:
            eval_path: Path to evaluation summary JSON
            analysis_path: Path to game analysis JSON
            probes_path: Path to probes results JSON
            out_dir: Output directory
            
        Returns:
            Dictionary with report data
        """
        # Load all data
        eval_data = self._load_json(eval_path)
        analysis_data = self._load_json(analysis_path)
        probes_data = self._load_json(probes_path)
        
        # Build report
        report = {
            "model": eval_data.get("model", "unknown"),
            "capabilities": self._build_capabilities(eval_data, analysis_data, probes_data),
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
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _build_capabilities(self, eval_data: Dict, analysis_data: Dict, probes_data: Dict) -> Dict:
        """Build capabilities table (C1-C14)."""
        
        capabilities = {}
        
        # C1-C3: From P1 probe
        if "P1" in probes_data:
            p1 = probes_data["P1"]
            capabilities["C1"] = {
                "description": "Légalité conditionnelle (top-1/top-3/fallback)",
                "top1_legal_pct": p1["metrics"]["total"]["top1_legal_pct"],
                "top3_legal_pct": p1["metrics"]["total"]["top3_legal_pct"],
                "fallback_pct": p1["metrics"]["total"]["fallback_pct"],
            }
            
            capabilities["C2"] = {
                "description": "Taxonomie des illégaux",
                "by_bucket": p1["by_bucket"],  # Will need to process
            }
            
            capabilities["C3"] = {
                "description": "Masse sur les coups légaux",
                "mass_legal_avg": p1["metrics"]["total"]["mass_legal_avg"],
                "mass_top1_avg": p1["metrics"]["total"]["mass_top1_avg"],
                "mass_top3_avg": p1["metrics"]["total"]["mass_top3_avg"],
            }
        
        # C4: From evaluation
        if "per_level" in eval_data:
            avg_score = eval_data.get("mean_score", 0)
            capabilities["C4"] = {
                "description": "Force brute (score vs moteur)",
                "mean_score": avg_score,
                "by_level": eval_data.get("per_level", {}),
            }
        
        # C5: From P2 probe
        if "P2" in probes_data:
            p2 = probes_data["P2"]
            capabilities["C5"] = {
                "description": "Mat en 1",
                "top1_mate_pct": p2["metrics"]["top1_mate_pct"],
                "top3_mate_pct": p2["metrics"]["top3_mate_pct"],
                "check_not_mate_count": p2["metrics"]["check_not_mate_count"],
            }
        
        # C6: From P3 probe
        if "P3" in probes_data:
            p3 = probes_data["P3"]
            capabilities["C6"] = {
                "description": "Captures",
                "take_winning_pct": p3["metrics"]["take_winning_capture_pct"],
                "take_losing_pct": p3["metrics"]["take_losing_capture_pct"],
                "take_free_pct": p3["metrics"]["take_free_piece_pct"],
            }
        
        # C7: From P4 probe
        if "P4" in probes_data:
            p4 = probes_data["P4"]
            capabilities["C7"] = {
                "description": "Anti-hang (perte matériel)",
                "hang_1pt_pct": p4["metrics"]["hang_1pt_pct"],
                "hang_3pt_pct": p4["metrics"]["hang_3pt_pct"],
            }
        
        # C8: From analysis
        if "C8_conversion" in analysis_data:
            c8 = analysis_data["C8_conversion"]
            capabilities["C8"] = {
                "description": "Conversion de l'avantage",
                "by_threshold": c8,
            }
        
        # C9: From endgame probe (not yet implemented)
        # C10: From promotion probe (not yet implemented)
        # C11: From analysis
        if "C11_repetitions" in analysis_data:
            c11 = analysis_data["C11_repetitions"]
            capabilities["C11"] = {
                "description": "Shuffling/répétition",
                "max_repetitions": c11["max"],
                "mean_repetitions": c11["mean"],
                "count_ge3": c11["count_ge3"],
            }
        
        # C12: From analysis
        if "C12_variance" in analysis_data:
            c12 = analysis_data["C12_variance"]
            capabilities["C12"] = {
                "description": "Variance & robustesse",
                "mean_score": c12["mean_score"],
                "ci_95": c12["ci_95"],
                "best_openings": c12["best_openings"],
                "worst_openings": c12["worst_openings"],
            }
        
        # C13-C14: Not yet implemented
        capabilities["C13"] = {"description": "Généralité next-move", "status": "NOT IMPLEMENTED"}
        capabilities["C14"] = {"description": "Stabilité fin de contexte", "status": "NOT IMPLEMENTED"}
        
        return capabilities
    
    def _identify_weaknesses(self, eval_data: Dict, analysis_data: Dict, probes_data: Dict) -> list:
        """Identify top weaknesses from the data."""
        
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
        if "P1" in probes_data:
            p1 = probes_data["P1"]
            fallback = p1["metrics"]["total"].get("fallback_pct", 0)
            if fallback > 5:
                weaknesses.append({
                    "id": "C1_high_fallback",
                    "description": f"Taux de fallback élevé: {fallback:.1f}%",
                    "impact": "Le modèle propose souvent des coups illégaux",
                    "estimated_elo_loss": f"~{fallback*2} Elo",
                })
        
        # C5: Low mate-in-1 detection
        if "P2" in probes_data:
            p2 = probes_data["P2"]
            top1_mate = p2["metrics"].get("top1_mate_pct", 0)
            if top1_mate < 50:
                weaknesses.append({
                    "id": "C5_low_mate_detection",
                    "description": f"Détection mat en 1 faible: {top1_mate:.1f}%",
                    "impact": "Le modèle ne voit pas les mats simples",
                    "estimated_elo_loss": "~50-100 Elo",
                })
        
        # C6: Poor capture decisions
        if "P3" in probes_data:
            p3 = probes_data["P3"]
            take_winning = p3["metrics"].get("take_winning_capture_pct", 0)
            take_losing = p3["metrics"].get("take_losing_capture_pct", 0)
            
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
        if "P4" in probes_data:
            p4 = probes_data["P4"]
            hang_3pt = p4["metrics"].get("hang_3pt_pct", 0)
            if hang_3pt > 5:
                weaknesses.append({
                    "id": "C7_high_hang_rate",
                    "description": f"Taux de hang ≥3pts: {hang_3pt:.1f}%",
                    "impact": "Le modèle abandonne du matériel gratuitement",
                    "estimated_elo_loss": f"~{hang_3pt*3} Elo",
                })
        
        # C8: Poor conversion
        if "C8_conversion" in analysis_data:
            c8 = analysis_data["C8_conversion"]
            for threshold, stats in c8.items():
                conversion = stats.get("conversion_rate", 0)
                if conversion < 0.7:  # Should win 70%+ from +2
                    weaknesses.append({
                        "id": f"C8_poor_conversion_{threshold}",
                        "description": f"Conversion faible depuis +{threshold}: {conversion*100:.1f}%",
                        "impact": "Ne sait pas convertir l'avantage",
                        "estimated_elo_loss": "~20-40 Elo",
                    })
        
        # Sort by estimated Elo loss
        weaknesses.sort(key=lambda x: float(x["estimated_elo_loss"].replace("~", "").replace(" Elo", "").split("-")[0]), reverse=True)
        
        return weaknesses[:5]  # Top 5
    
    def _generate_recommendations(self, eval_data: Dict, analysis_data: Dict, probes_data: Dict) -> list:
        """Generate training recommendations based on weaknesses."""
        
        recommendations = []
        
        # Based on C4 (score vs engine)
        if "mean_score" in eval_data:
            score = eval_data["mean_score"]
            if score < 0.5:
                recommendations.append({
                    "priority": "HIGH",
                    "action": "Fine-tune on engine games (levels 0-3)",
                    "objective": "Improve score from {score:.3f} to >0.6",
                    "method": "SFT with engine-labeled moves",
                    "estimated_gain": "+100-200 Elo",
                })
        
        # Based on C1 (legality)
        if "P1" in probes_data:
            p1 = probes_data["P1"]
            fallback = p1["metrics"]["total"].get("fallback_pct", 0)
            if fallback > 3:
                recommendations.append({
                    "priority": "HIGH",
                    "action": "Legal finetuning with legality loss",
                    "objective": f"Reduce fallback rate from {fallback:.1f}% to <2%",
                    "method": "Train on on-policy positions with legality mask",
                    "estimated_gain": "+50-100 Elo",
                })
        
        # Based on C5 (mate in 1)
        if "P2" in probes_data:
            p2 = probes_data["P2"]
            top1_mate = p2["metrics"].get("top1_mate_pct", 0)
            if top1_mate < 70:
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": "Mate-in-1 finetuning",
                    "objective": f"Increase mate-in-1 detection from {top1_mate:.1f}% to >80%",
                    "method": "SFT on mate-in-1 positions with high weight",
                    "estimated_gain": "+30-50 Elo",
                })
        
        # Based on C6 (captures)
        if "P3" in probes_data:
            p3 = probes_data["P3"]
            take_winning = p3["metrics"].get("take_winning_capture_pct", 0)
            if take_winning < 80:
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": "Capture finetuning with SEE labels",
                    "objective": f"Increase winning capture rate from {take_winning:.1f}% to >85%",
                    "method": "SFT with SEE-weighted capture loss",
                    "estimated_gain": "+20-40 Elo",
                })
        
        # Based on C7 (hanging)
        if "P4" in probes_data:
            p4 = probes_data["P4"]
            hang_3pt = p4["metrics"].get("hang_3pt_pct", 0)
            if hang_3pt > 3:
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": "Anti-hang finetuning",
                    "objective": f"Reduce hang rate from {hang_3pt:.1f}% to <2%",
                    "method": "RL with material loss penalty",
                    "estimated_gain": "+20-30 Elo",
                })
        
        # Based on C8 (conversion)
        if "C8_conversion" in analysis_data:
            c8 = analysis_data["C8_conversion"]
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
                elif "mean_score" in cap:
                    val = f"{cap['mean_score']:.3f}"
                elif "by_threshold" in cap:
                    val = ", ".join(f"+{k}: {v['conversion_rate']*100:.1f}%" for k, v in cap["by_threshold"].items())
                else:
                    val = json.dumps(cap, indent=0)[:50]
                
                md.append(f"| {cid} | {cap['description']} | - | {val} |")
        
        md.append("")
        
        # Top weaknesses
        md.append("## 🎯 Top 5 Weaknesses")
        md.append("")
        
        for i, weakness in enumerate(report["top_weaknesses"], 1):
            md.append(f"### {i}. {weakness['description']}")
            md.append(f"- **Impact:** {weakness['impact']}")
            md.append(f"- **Estimated Elo loss:** {weakness['estimated_elo_loss']}")
            md.append("")
        
        # Recommendations
        md.append("## 🚀 Training Recommendations")
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
                        help="Path to evaluation summary JSON")
    parser.add_argument("--analysis", type=str, required=True,
                        help="Path to game analysis JSON")
    parser.add_argument("--probes", type=str, required=True,
                        help="Path to probes results JSON")
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
