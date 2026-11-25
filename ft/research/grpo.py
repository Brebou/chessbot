"""j-batch: RLOO/group-relative game-level fine-tuning (GRPO-flavored).

Motivation (ledger run13 Arm 2): outcome-weighted rows (REINFORCE-lite)
gave ZERO because the positive leg had N~35 informative games out of 2400
(outcome vs the class engine is nearly constant: ~1.7% wins). The
group-relative baseline converts EVERY game into gradient: within the K
rollouts of one (opening, level, color) group,
A_i = r_i - mean(r of the other K-1 games)   (RLOO; lower variance than
mean/std for small K). "Punish the buildup, not just the blunder" is then
captured without its known degeneracy (whole-game pull-down that pushes the
model to avoid openings / lose fast):

  - positive-advantage games ONLY push UP the moves actually played
    (self-distillation toward the relatively-better lines of the SAME
    group) - the f-batch anti-collapse leg at ~50% coverage instead of ~2%;
  - negative-advantage games contribute nothing (j1, pure RLOO) or SF
    corrections on their blunder states weighted by |A| (j2, hybrid with
    the one ingredient that kept proxy alive in the f-batch);
  - the KL anchor (0.1) to base stands guard same as everywhere.

Collection subtlety the campaign's greedy collector cannot handle: greedy
decoding + deterministic engine => IDENTICAL games within a group => zero
variance => zero advantage. Games are sampled here with a legal-masked
temperature rollout (eval stays greedy - unchanged, paired, seeded).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import chess

from chess_hw.match import (EnginePlayer, load_openings, play_game,
                            score_for)
from chess_hw.model import load_model
from chess_hw.play import MoveChoice
from diag.fast_probe import FastProbePlayer
from ft.common.serialize import jsonl_write

EVAL_LEVELS = (1, 2, 3)
MAX_PLIES = 250
RESULTS = Path("ft/research/results.jsonl")


class SamplingProbe(FastProbePlayer):
    """Legal-masked temperature-sampling player (drop-in for ModelPlayer).

    Inherits the batched last-position-only forward; overrides choose() to
    sample instead of greedy top-3. Deterministic given (seed, board) so
    groups are reproducible.
    """

    def __init__(self, model, tokenizer, *, temperature: float = 0.9,
                 seed: int = 0, **kw):
        super().__init__(model, tokenizer, **kw)
        self.temperature = max(float(temperature), 1e-3)
        self.rng = random.Random(seed)
        self.name = f"grpo-sample-T{self.temperature}"
        self._uci2id = {t: i for i, t in enumerate(self._id2tok)}

    @torch.inference_mode()
    def choose(self, board: chess.Board) -> MoveChoice:
        logits = self._forward_auto(
            [[m.uci() for m in board.move_stack]])[0].float().cpu().numpy()
        legal_ids = []
        for mv in board.legal_moves:
            tid = self._uci2id.get(mv.uci())
            if tid is not None and tid < logits.shape[0]:
                legal_ids.append(tid)
        if not legal_ids:                      # vocab gap: random legal move
            mvs = list(board.legal_moves)
            mv = self.rng.choice(mvs) if mvs else None
            return MoveChoice(mv, 0)
        masked = np.full(logits.shape[0], -1e9, dtype=np.float32)
        masked[legal_ids] = logits[legal_ids] / self.temperature
        p = np.exp(masked - masked.max())
        p /= p.sum()
        tid = self.rng.choices(range(logits.shape[0]), weights=p.tolist(),
                               k=1)[0]
        return MoveChoice(board.parse_uci(self._id2tok[tid]), 1)


def rloo_advantages(rewards: Sequence[float]) -> List[float]:
    """A_i = r_i - mean(r over the OTHER games). Constant groups -> 0."""
    n = len(rewards)
    if n <= 1:
        return [0.0 for _ in rewards]
    total = sum(rewards)
    return [r - (total - r) / (n - 1) for r in rewards]


def group_advantages(group: dict) -> List[float]:
    return rloo_advantages([g["out"] for g in group["games"]])


def collect_groups(policy_path: str, oids: List[int], levels: List[int], *,
                   k_rollouts: int, temperature: float, device: str = "cpu",
                   seed_tag: str = "grpo", max_plies: int = MAX_PLIES,
                   progress: Optional[object] = None) -> List[dict]:
    """K sampled rollouts per (oid, level, color) group vs the class engine.

    Returns groups: {oid, lv, color(=model white), games:[{out, states}]}
    where states carry fen/history/model_move as in the REINFORCE collector.
    """
    from chess_hw import ChessTokenizer

    model = load_model(policy_path, device=device)
    probe = SamplingProbe(model, ChessTokenizer(), temperature=temperature,
                          seed=0, batch_size=256)
    book = load_openings()
    groups: List[dict] = []
    for n, oid in enumerate(oids):
        opening = book[oid]
        for lv in levels:
            for white_is_model in (True, False):
                games = []
                for kg in range(k_rollouts):
                    probe.rng = random.Random(
                        f"{seed_tag}-{oid}-{lv}-{white_is_model}-{kg}")
                    eng = EnginePlayer(lv, seed=f"{seed_tag}-{oid}-{lv}")
                    if white_is_model:
                        rec = play_game(probe, eng, opening, oid,
                                        max_plies=max_plies)
                    else:
                        rec = play_game(eng, probe, opening, oid,
                                        max_plies=max_plies)
                    board = chess.Board()
                    for u in opening:
                        board.push_uci(u)
                    hist = list(opening)
                    start = len(opening)
                    mstates = []
                    for i in range(start, len(rec.moves)):
                        is_model_ply = ((i - start) % 2 == 0) == white_is_model
                        if is_model_ply:
                            mstates.append({"fen": board.fen(),
                                            "history": list(hist),
                                            "model_move": rec.moves[i]})
                        board.push_uci(rec.moves[i])
                        hist.append(rec.moves[i])
                    result = rec.result
                    color = chess.WHITE if white_is_model else chess.BLACK
                    games.append({"out": score_for(result, color),
                                  "states": mstates})
                groups.append({"oid": oid, "lv": lv,
                               "color": "white" if white_is_model else "black",
                               "games": games})
        if progress is not None and hasattr(progress, "show"):
            progress.show(f"  grpo groups {n+1}/{len(oids)} "
                          f"({(n+1)*len(levels)*2} groups)")
    return groups


def build_grpo_mix(groups: List[dict], out_path: Path, *,
                   engine_path: str = "", sf_depth: int = 8,
                   sf_workers: int = 8, w_pos: float = 1.0,
                   w_neg: float = 0.0, cap: int = 20000) -> tuple:
    """Rows: for every group, advantage-normalized weights.

    A_i range-normalized by the group's max|A| (robust for tiny groups):
    positive games -> own-move rows, weight w_pos * Ahat;
    negative games (w_neg > 0, i.e. j2) -> SF corrections on their blunder
    states with weight w_neg * |Ahat| (non-blunders: own-move rehearsal at
    0.25 x that, same shape as the REINFORCE mix).
    """
    rows: List[dict] = []
    neg_states: List[tuple] = []
    for grp in groups:
        A = group_advantages(grp)
        amax = max((abs(a) for a in A), default=0.0)
        if amax <= 0.0:
            continue
        for game, a in zip(grp["games"], A):
            ahat = a / amax
            if ahat > 0 and w_pos > 0:
                for s in game["states"]:
                    rows.append({"bucket": "normal", "key": s["fen"],
                                 "start_fen": s["fen"],
                                 "history": s["history"],
                                 "target": s["model_move"],
                                 "weight": round(w_pos * ahat, 4)})
            elif ahat < 0 and w_neg > 0:
                wl = w_neg * (-ahat)
                for s in game["states"]:
                    neg_states.append((s["fen"], s["history"],
                                       s["model_move"], wl))
    if neg_states:
        from ft.sft.poc_loop import poc_classify, poc_score
        from ft.common.board_io import n_pieces_from_fen
        uniq: Dict[str, tuple] = {}
        for fen, hist, mv, _wl in neg_states:
            if fen not in uniq:
                uniq[fen] = (hist, mv)
        print(f"grpo: scoring {len(uniq)} unique negative-advantage "
              "states...", flush=True)
        scores = poc_score([(fen, mv) for fen, (_, mv) in uniq.items()],
                           engine_path=engine_path, depth=sf_depth,
                           workers=sf_workers)
        n_corr = 0
        for fen, (hist, mv) in uniq.items():
            wl = min(w for f, h, m, w in neg_states if f == fen)
            sc = scores.get(fen, {})
            blunder = poc_classify(sc.get("cp_best"), sc.get("cp_model"),
                                   sc.get("mate_best"), sc.get("mate_model"),
                                   n_pieces_from_fen(fen))
            if blunder:
                target = sc.get("best_uci")
                if not target or target == mv:
                    continue
                rows.append({"bucket": "normal", "key": fen,
                             "start_fen": fen, "history": hist,
                             "target": target, "weight": round(wl, 4)})
                n_corr += 1
            else:
                rows.append({"bucket": "normal", "key": fen,
                             "start_fen": fen, "history": hist,
                             "target": mv, "weight": round(0.25 * wl, 4)})
        print(f"grpo: {n_corr} corrections from {len(uniq)} negative states",
              flush=True)
    rng = random.Random(42)
    rng.shuffle(rows)
    if cap and len(rows) > cap:
        rows = rows[:cap]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_write(out_path, rows)
    return len(rows), rows


def run_grpo_arm(out_dir: Path, *, model_path: str, base_path: str,
                 cfg: dict, eval_openings: int, selfplay_games: int,
                 eval_workers: int, eval_seed: int, sf_depth: int,
                 sf_workers: int, engine_path: str,
                 records_cache: dict) -> dict:
    from ft.research.sweep import train_candidate, resolve_engine
    from ft.research.arc2 import _select_oids
    from ft.eval.paired_eval import paired_engine_eval, selfplay_paired
    name = cfg["name"]
    t0 = time.monotonic()
    if engine_path:
        engine_path = resolve_engine(engine_path)
    oids = _select_oids(cfg.get("seed_tag", f"grpo-{name}"),
                        cfg.get("n_openings", 200))
    groups = collect_groups(model_path, oids, cfg.get("levels", [1, 2]),
                            k_rollouts=cfg.get("k_rollouts", 8),
                            temperature=cfg.get("temperature", 0.9),
                            device=cfg.get("device", "cpu"),
                            seed_tag=f"grpo-{name}")
    mix_path = out_dir / f"mix_{name}.jsonl"
    n_rows, _rows = build_grpo_mix(
        groups, mix_path, engine_path=engine_path,
        sf_depth=cfg.get("sf_depth", sf_depth), sf_workers=sf_workers,
        w_pos=cfg.get("w_pos", 1.0), w_neg=cfg.get("w_neg", 0.0),
        cap=cfg.get("cap", 20000))
    pos_groups = sum(1 for g in groups
                     if any(a > 0 for a in group_advantages(g)))
    print(f"grpo: {len(groups)} groups ({pos_groups} with signal), "
          f"{n_rows} rows [{time.monotonic()-t0:.0f}s]", flush=True)
    done = train_candidate(
        mix_path, model_path=model_path, base_path=base_path,
        out_dir=out_dir / name, epochs=cfg.get("epochs", 1),
        lr=cfg["lr"], kl=cfg["kl"], batch=cfg.get("batch", 512), seed=42,
        eval_openings=eval_openings, selfplay_games=selfplay_games,
        eval_workers=eval_workers, records_cache=records_cache,
        device=cfg.get("device", "cpu"), eval_seed=eval_seed)
    mpath = done.get("final_model", str(out_dir / name / "final" /
                                        "model.safetensors"))
    p = paired_engine_eval(mpath, base_path, levels=EVAL_LEVELS,
                           n_openings=eval_openings, seed=eval_seed,
                           workers=eval_workers, n_bootstrap=1000,
                           records_cache=records_cache)
    sl = selfplay_paired(mpath, base_path, n_games=selfplay_games,
                         seed=eval_seed, workers=eval_workers)
    row = {"run": f"grpo/{name}", "config": cfg, "mix_rows": n_rows,
           "groups": len(groups), "groups_with_signal": pos_groups,
           "engine_delta": p["delta"], "sigma": p["sigma"],
           "per_level": p["per_level_delta"],
           "proxy_delta": sl["proxy_delta"],
           "balanced": (p["delta"] + sl["proxy_delta"]) / 2.0,
           "eval_seed": eval_seed, "eval_openings": eval_openings,
           "elapsed_s": round(time.monotonic() - t0)}
    print("GRPO:", json.dumps(row), flush=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


GRPO_CONFIGS: Dict[str, dict] = {
    # j1: pure RLOO group-relative, NO SF in the loss at all (only the eval).
    # Measures the group-relative axis cleanly.
    "j1": {"n_openings": 200, "k_rollouts": 8, "temperature": 0.9,
           "levels": [1, 2], "w_pos": 1.0, "w_neg": 0.0, "cap": 20000,
           "epochs": 1, "kl": 0.1, "lr": 5e-5, "batch": 512,
           "seed_tag": "grpo-j1"},
    # j2: + corrections on negative-advantage games' blunders, weight |Ahat|.
    "j2": {"n_openings": 200, "k_rollouts": 8, "temperature": 0.9,
           "levels": [1, 2], "w_pos": 1.0, "w_neg": 1.5, "cap": 20000,
           "epochs": 1, "kl": 0.1, "lr": 5e-5, "batch": 512,
           "sf_depth": 8, "seed_tag": "grpo-j2"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--anchor-path", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--configs", default="j1", help="comma-separated: j1,j2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--k-rollouts", type=int, default=None)
    ap.add_argument("--n-openings", type=int, default=None)
    ap.add_argument("--sf-depth", type=int, default=8)
    ap.add_argument("--sf-workers", type=int, default=12)
    ap.add_argument("--engine", default="")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--eval-openings", type=int, default=10)
    ap.add_argument("--selfplay-games", type=int, default=20)
    ap.add_argument("--eval-workers", type=int, default=22)
    ap.add_argument("--eval-seed", type=int, default=1234)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.anchor_path or args.model_path
    records_cache: dict = {}
    for name in [c for c in args.configs.split(",") if c]:
        if name not in GRPO_CONFIGS:
            raise SystemExit(f"unknown grpo config '{name}'; known: "
                             f"{', '.join(GRPO_CONFIGS)}")
        cfg = dict(GRPO_CONFIGS[name])
        cfg["name"] = name
        cfg["device"] = args.device
        for key, val in (("temperature", args.temperature),
                         ("k_rollouts", args.k_rollouts),
                         ("n_openings", args.n_openings),
                         ("batch", args.batch), ("lr", args.lr),
                         ("epochs", args.epochs), ("cap", args.cap)):
            if val is not None:
                cfg[key] = val
        run_grpo_arm(out_dir, model_path=args.model_path,
                     base_path=base_path, cfg=cfg,
                     eval_openings=args.eval_openings,
                     selfplay_games=args.selfplay_games,
                     eval_workers=args.eval_workers,
                     eval_seed=args.eval_seed, sf_depth=args.sf_depth,
                     sf_workers=args.sf_workers, engine_path=args.engine,
                     records_cache=records_cache)


if __name__ == "__main__":
    main()
