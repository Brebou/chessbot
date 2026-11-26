"""Add legal-move labels (no Stockfish) to existing shards, in place.
  python -m plan.addlegal data/r1/shards data/r2/shards
"""
import glob, multiprocessing as mp, sys, torch
from plan.common import legal_arrays, load_vocab


def _one(f):
    v = load_vocab("base_model"); games = torch.load(f, weights_only=False)
    for g in games:
        if "lg_flat" not in g:
            g["lg_flat"], g["lg_off"] = legal_arrays([v.id2move[int(i)] for i in g["ids"]], v)
    torch.save(games, f); return f


if __name__ == "__main__":
    files = [f for d in sys.argv[1:] for f in sorted(glob.glob(f"{d}/shard_*.pt"))]
    with mp.get_context("spawn").Pool(min(22, len(files))) as p:
        for f in p.imap_unordered(_one, files):
            print("done", f, flush=True)
