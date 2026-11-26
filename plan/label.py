"""Turn games into training shards.

  sf mode       : Stockfish multipv soft targets on every ply (the core of the plan)
  winmoves mode : one-hot on OUR moves in won games (for beam.py output)
  legal mode    : no Stockfish, only legal-move labels (w=0): free data for the legality loss

  python -m plan.label --games data/r1/games.jsonl --out data/r1/shards --mode sf --depth 8
  python -m plan.label --games data/beam/wins.jsonl --out data/beam/shards --mode winmoves

Shard = list of dicts: ids[k] int16, idx[k,K] int16 (-1 pad), p[k,K], cp[k,K], w[k], ours[k].
Row t describes the position AFTER t moves (input position t predicts move t+1).
"""
import argparse, json, multiprocessing as mp, os, time
import numpy as np, torch, chess, chess.engine
from plan.common import legal_arrays, load_vocab, open_sf

_S = {}
MAXK = 255


def _init(model_dir, depth, multipv, temp, skip, oppw, mode, sf_path):
    _S.update(v=load_vocab(model_dir), depth=depth, K=multipv, temp=temp, skip=skip, oppw=oppw, mode=mode)
    _S["sf"] = open_sf(sf_path) if mode == "sf" else None


def _label(g):
    v, K = _S["v"], _S["K"]
    moves = g["moves"][:MAXK]; k = len(moves)
    ids = np.array([v.move2id[m] for m in moves], np.int16)
    idx = -np.ones((k, K), np.int16); p = np.zeros((k, K), np.float16)
    cp = np.zeros((k, K), np.float16); w = np.zeros(k, np.float16); ours = np.zeros(k, bool)
    board = chess.Board()
    for t, u in enumerate(moves):
        white = board.turn == chess.WHITE
        o = g["ours"] == "both" or (g["ours"] == "w") == white
        ours[t] = o
        if _S["mode"] == "winmoves":
            if o:
                idx[t, 0] = ids[t]; p[t, 0] = 1; w[t] = 1
        elif _S["mode"] == "sf" and t >= _S["skip"] and (o or _S["oppw"] > 0) and board.legal_moves.count() > 1:
            info = _S["sf"].analyse(board, chess.engine.Limit(depth=_S["depth"]), multipv=K)
            mv, sc = [], []
            for i in info:
                mv.append(v.move2id.get(i["pv"][0].uci(), -1))
                sc.append(i["score"].pov(board.turn).score(mate_score=10000))
            keep = [j for j, m in enumerate(mv) if m >= 0]
            if keep:
                sc_a = np.array([sc[j] for j in keep], np.float64)
                z = np.clip(sc_a, -1000, 1000) / _S["temp"]; z -= z.max()
                pr = np.exp(z); pr /= pr.sum()
                n = len(keep)
                idx[t, :n] = [mv[j] for j in keep]; p[t, :n] = pr; cp[t, :n] = sc_a
                w[t] = 1.0 if o else _S["oppw"]
        board.push_uci(u)
    lf, lo = legal_arrays(moves, v)
    return dict(ids=ids, idx=idx, p=p, cp=cp, w=w, ours=ours, lg_flat=lf, lg_off=lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["sf", "winmoves", "legal"], default="sf")
    ap.add_argument("--model-dir", default="base_model", help="only for the vocab")
    ap.add_argument("--depth", type=int, default=8); ap.add_argument("--multipv", type=int, default=5)
    ap.add_argument("--temp", type=float, default=60.0, help="centipawn softmax temperature")
    ap.add_argument("--skip", type=int, default=0, help="skip first N plies (book)")
    ap.add_argument("--opp-weight", type=float, default=0.5, help="loss weight on opponent-move positions")
    ap.add_argument("--workers", type=int, default=22); ap.add_argument("--shard-games", type=int, default=2000)
    ap.add_argument("--keep-draws", action="store_true"); ap.add_argument("--limit", type=int)
    ap.add_argument("--sf")
    a = ap.parse_args()
    games = [json.loads(l) for f in a.games for l in open(f)]
    if a.mode == "winmoves":
        games = [g for g in games if g["result"] == 1 or (a.keep_draws and g["result"] == .5)]
    if a.limit:
        games = games[:a.limit]
    print(f"{len(games)} games -> {a.out} ({a.mode})")
    os.makedirs(a.out, exist_ok=True)
    t0, buf, sh = time.time(), [], 0
    with mp.get_context("spawn").Pool(a.workers, _init, (a.model_dir, a.depth, a.multipv, a.temp, a.skip,
                                                         a.opp_weight, a.mode, a.sf)) as pool:
        for i, r in enumerate(pool.imap(_label, games, chunksize=4)):
            buf.append(r)
            if len(buf) >= a.shard_games:
                torch.save(buf, f"{a.out}/shard_{sh:04d}.pt"); sh += 1; buf = []
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(games)}  {(i+1)/(time.time()-t0):.1f} games/s", flush=True)
    if buf:
        torch.save(buf, f"{a.out}/shard_{sh:04d}.pt")


if __name__ == "__main__":
    main()
