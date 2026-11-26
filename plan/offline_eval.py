"""Dense, low-noise dev metric: how good are the model's DECODED moves on held-out positions?
Positions come from shards labelled by label.py --mode sf (use a held-out collection, e.g. other seed).

  python -m plan.offline_eval --model out/r1/last --shards data/heldout/shards --max-games 600
  python -m plan.offline_eval --model base_model  --shards data/heldout/shards --max-games 600
cp loss is a LOWER BOUND when the decoded move is outside Stockfish's multipv list.
"""
import argparse, glob, numpy as np, torch, chess
from plan.common import load_model, logits_of


@torch.no_grad()
def evaluate(model, vocab, games, dev):
    R = dict(n=0, top1=0, intop=0, randmv=0, fallback=0, loss=[], blunder=0, mate_avail=0, mate_hit=0)
    for g in games:
        ids = torch.from_numpy(g["ids"].astype(np.int64)); k = len(ids)
        X = torch.cat([torch.tensor([vocab.bos]), ids[:-1]]).unsqueeze(0).to(dev)
        top3 = torch.topk(logits_of(model, X)[0], 3, -1).indices.cpu().tolist()
        board = chess.Board()
        for t in range(k):
            u = vocab.id2move[int(ids[t])]
            if g["ours"][t] and g["idx"][t, 0] >= 0:
                legal = {m.uci() for m in board.legal_moves}
                ch, rank = None, 3
                for r, tok in enumerate(top3[t]):
                    mv = vocab.id2move.get(tok)
                    if mv in legal:
                        ch, rank = tok, r; break
                valid = g["idx"][t] >= 0
                cps = g["cp"][t][valid].astype(np.float64); idxs = g["idx"][t][valid]
                best = cps.max(); R["n"] += 1
                if ch is None:
                    R["randmv"] += 1; loss = 1000.0  # random legal move: treated as a blunder
                else:
                    R["fallback"] += rank > 0
                    R["top1"] += ch == idxs[0]; R["intop"] += ch in idxs
                    loss = best - cps[list(idxs).index(ch)] if ch in idxs else best - cps.min()
                loss = float(min(max(loss, 0), 1000)); R["loss"].append(loss); R["blunder"] += loss >= 200
                if best >= 9000:
                    R["mate_avail"] += 1; R["mate_hit"] += ch is not None and loss < 1
            board.push_uci(u)
    n = max(R["n"], 1)
    return {"positions": R["n"], "sf_top1_match": R["top1"] / n, "in_sf_top5": R["intop"] / n,
            "mean_cp_loss_lb": float(np.mean(R["loss"])), "blunder_rate(>=200cp)": R["blunder"] / n,
            "random_move_rate": R["randmv"] / n, "top1_illegal_rate": R["fallback"] / n,
            "mate_hit": R["mate_hit"] / max(R["mate_avail"], 1), "mate_positions": R["mate_avail"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--shards", required=True)
    ap.add_argument("--max-games", type=int, default=600); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    model, vocab = load_model(a.model, a.device)
    games = []
    for f in sorted(glob.glob(f"{a.shards}/shard_*.pt")):
        games += torch.load(f, weights_only=False)
        if len(games) >= a.max_games:
            break
    for k, v in evaluate(model, vocab, games[:a.max_games], a.device).items():
        print(f"{k:>24}: {v:.4f}" if isinstance(v, float) else f"{k:>24}: {v}")
