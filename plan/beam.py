"""Step 2: search over OUR top-3 legal moves against the deterministic engine to find WINNING lines.
Output games feed `label.py --mode winmoves`. Needs the OFFICIAL engine in engine_hook.py to be meaningful.

  python -m plan.beam --model out/r1/last --levels 2 3 --n-openings 1500 --seed 11 \
       --width 8 --sf-depth 6 --out data/beam/wins.jsonl
"""
import argparse, json, multiprocessing as mp, torch, chess, chess.engine
from plan.common import (engine_move, load_model, load_openings, logits_of, open_sf, status)

_G = {}


def _init(model, device):
    torch.set_num_threads(1)
    _G["m"], _G["v"] = load_model(model, device); _G["dev"] = device; _G["sf"] = open_sf()


@torch.no_grad()
def search(opening, our, level, W, sfd, keep, keep_draws):
    m, v, dev, sf = _G["m"], _G["v"], _G["dev"], _G["sf"]
    b0 = chess.Board(); h0 = []
    for u in opening:
        b0.push_uci(u); h0.append(v.move2id[u])

    def settle(b, h):
        while True:
            st = status(b, our)
            if st or b.turn == our:
                return st
            mv = engine_move(b, level); b.push(mv); h.append(v.move2id.get(mv.uci(), 0))

    wins, draws, beam = [], [], []
    st = settle(b0, h0)
    if not st:
        beam = [(b0, h0)]
    while beam:
        cand = []
        for b, h in beam:
            lg = logits_of(m, torch.tensor([[v.bos] + h], device=dev))[0, -1]
            legal = {x.uci() for x in b.legal_moves}
            picks = [v.id2move[t] for t in torch.topk(lg, 12).indices.tolist()
                     if v.id2move.get(t) in legal][:3]
            for u in picks:
                nb, nh = b.copy(), list(h); nb.push_uci(u); nh.append(v.move2id[u])
                st = settle(nb, nh)
                if st:
                    line = [x.uci() for x in nb.move_stack]
                    (wins if st[0] == 1 else draws if st[0] == .5 else []).append(line)
                else:
                    sc = sf.analyse(nb, chess.engine.Limit(depth=sfd))["score"].pov(our).score(mate_score=10000)
                    cand.append((sc, nb.epd(), nb, nh))
        if wins:
            break
        cand.sort(key=lambda x: -x[0]); seen, beam = set(), []
        for sc, e, nb, nh in cand:
            if e not in seen:
                seen.add(e); beam.append((nb, nh))
            if len(beam) >= W:
                break
    pool = sorted(wins, key=len) or (sorted(draws, key=len) if keep_draws else [])
    return [dict(moves=l, opening=list(opening), ours="w" if our == chess.WHITE else "b",
                 opp=f"L{level}", result=1.0 if wins else 0.5, reason="beam") for l in pool[:keep]]


def _run(t):
    return search(*t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--openings"); ap.add_argument("--n-openings", type=int, default=500)
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--random-plies", type=int, default=6)
    ap.add_argument("--width", type=int, default=8); ap.add_argument("--sf-depth", type=int, default=6)
    ap.add_argument("--keep", type=int, default=1); ap.add_argument("--keep-draws", action="store_true")
    ap.add_argument("--workers", type=int, default=22); ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ops = load_openings(a.openings, a.n_openings, a.seed, a.random_plies)
    tasks = [(op, c, l, a.width, a.sf_depth, a.keep, a.keep_draws)
             for l in a.levels for op in ops for c in (chess.WHITE, chess.BLACK)]
    import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    found = 0
    with mp.get_context("spawn").Pool(a.workers, _init, (a.model, a.device)) as pool, open(a.out, "w") as f:
        for i, res in enumerate(pool.imap_unordered(_run, tasks)):
            for g in res:
                f.write(json.dumps(g) + "\n"); found += 1
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(tasks)} searched, {found} winning/kept lines", flush=True)
