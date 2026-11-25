"""Play games (model vs engine levels / vs another model). Used for eval AND data collection.

  python -m plan.play --model base_model --opps L1 L2 L3 --n-openings 40 --seed 1234
  python -m plan.play --model out/r1/last --opps M:base_model --n-openings 200      # head-to-head proxy
  python -m plan.play --model base_model --opps L1 L2 L3 M:base_model --n-openings 3000 \
        --seed 1 --save data/r1/games.jsonl                                           # collection
Paired delta vs an earlier run:  --baseline results/base.jsonl
"""
import argparse, json, math, multiprocessing as mp
from collections import Counter, defaultdict

import chess, torch
from plan.common import (choose_move, engine_move, load_model, load_openings, status)

_G = {}


def _init(model_path, opp_models, device):
    torch.set_num_threads(1)
    _G["m"], _G["v"] = load_model(model_path, device)
    _G["opp"] = {p: load_model(p, device)[0] for p in opp_models}
    _G["dev"] = device


def play_game(model, vocab, device, opening, opp, our_color, opp_model=None):
    board = chess.Board(); hist = []
    for u in opening:
        board.push_uci(u); hist.append(vocab.move2id[u])
    fallbacks = 0
    while True:
        st = status(board, our_color)
        if st:
            result, reason = st; break
        if board.turn == our_color:
            u, r = choose_move(model, vocab, board, hist, device)
            if u is None:
                result, reason = 0.0, "forfeit"; break
            fallbacks += r > 0
        elif opp_model is not None:
            u, _ = choose_move(opp_model, vocab, board, hist, device)
            if u is None:
                result, reason = 1.0, "opp_forfeit"; break
        else:
            u = engine_move(board, int(opp[1:])).uci()
        board.push_uci(u); hist.append(vocab.move2id.get(u, 0))
    return dict(moves=[m.uci() for m in board.move_stack], opening=list(opening),
                ours="w" if our_color == chess.WHITE else "b", opp=opp,
                result=result, reason=reason, fallbacks=int(fallbacks))


def _run(task):
    opening, opp, color = task
    om = _G["opp"].get(opp[2:]) if opp.startswith("M:") else None
    return play_game(_G["m"], _G["v"], _G["dev"], opening, opp, color, om)


def run_games(model, opps, openings, workers, device="cpu"):
    tasks = [(op, o, c) for o in opps for op in openings for c in (chess.WHITE, chess.BLACK)]
    mp_models = [o[2:] for o in opps if o.startswith("M:")]
    with mp.get_context("spawn").Pool(workers, _init, (model, mp_models, device)) as pool:
        for i, g in enumerate(pool.imap(_run, tasks, chunksize=1)):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tasks)} games", flush=True)
            yield g


def summarize(games, baseline=None):
    by = defaultdict(list)
    for g in games:
        by[g["opp"]].append(g)
    tot = []
    for opp, gs in sorted(by.items()):
        sc = [g["result"] for g in gs]
        n = len(sc); mean = sum(sc) / n
        reasons = Counter((g["reason"], g["result"]) for g in gs)
        fb = sum(g["fallbacks"] for g in gs) / n
        print(f"{opp:>14}: score {mean:.3f} (W{sum(s==1 for s in sc)} D{sum(s==.5 for s in sc)} "
              f"L{sum(s==0 for s in sc)}, n={n})  fallbacks/game {fb:.1f}")
        print("                 endings:", dict(reasons))
        if opp.startswith("L"):
            tot.append(mean)
    if tot:
        print(f"official-style P = mean over levels = {sum(tot)/len(tot):.4f}")
    if baseline:
        key = lambda g: (g["opp"], tuple(g["opening"]), g["ours"])
        base = {key(g): g["result"] for g in map(json.loads, open(baseline))}
        d = [(g["opp"], g["result"] - base[key(g)]) for g in games if key(g) in base]
        for opp in sorted({o for o, _ in d}):
            x = [v for o, v in d if o == opp]; m = sum(x) / len(x)
            sd = math.sqrt(sum((v - m) ** 2 for v in x) / max(1, len(x) - 1))
            print(f"paired delta {opp:>14}: {m:+.3f} ± {sd/math.sqrt(len(x)):.3f} (se, n={len(x)})")
        lv = [o for o in sorted({o for o, _ in d}) if o.startswith("L")]
        if lv:
            ms = [sum(v for o, v in d if o == l) / sum(1 for o, _ in d if o == l) for l in lv]
            print(f"paired delta P (mean of levels): {sum(ms)/len(ms):+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--opps", nargs="+", default=["L1", "L2", "L3"])
    ap.add_argument("--openings"); ap.add_argument("--n-openings", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--random-plies", type=int, default=6)
    ap.add_argument("--workers", type=int, default=22); ap.add_argument("--device", default="cpu")
    ap.add_argument("--save"); ap.add_argument("--baseline")
    a = ap.parse_args()
    ops = load_openings(a.openings, a.n_openings, a.seed, a.random_plies)
    games = list(run_games(a.model, a.opps, ops, a.workers, a.device))
    if a.save:
        import os; os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        with open(a.save, "w") as f:
            for g in games:
                f.write(json.dumps(g) + "\n")
    summarize(games, a.baseline)
