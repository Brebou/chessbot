"""Run FIRST: python -m plan.selfcheck --model base_model   (checks every assumption of the pipeline)"""
import argparse, time, torch, chess
from plan.common import choose_move, engine_move, load_model, open_sf, status


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="base_model")
    a = ap.parse_args()
    model, v = load_model(a.model)
    print("model class:", type(model).__name__, "| params:", sum(p.numel() for p in model.parameters()))
    print("UCI tokens:", len(v.move2id), "| bos id:", v.bos)
    b = chess.Board(); missing = [m.uci() for m in b.legal_moves if m.uci() not in v.move2id]
    assert not missing, f"legal start moves not in vocab: {missing}"
    from safetensors import safe_open
    with safe_open(f"{a.model}/model.safetensors", "pt") as f:
        keys = set(f.keys())
    sd = set(model.state_dict().keys())
    print("keys in file but not state_dict:", sorted(keys - sd)[:5], "| reverse:", sorted(sd - keys)[:5])
    assert keys <= sd, "common.save_checkpoint will fail: adapt it"
    hist, t0 = [], time.time()
    for _ in range(40):
        u, r = choose_move(model, v, b, hist, "cpu")
        if u is None:
            print("no legal move in top-3 at ply", len(hist), "(judge would play a random legal move)"); break
        b.push_uci(u); hist.append(v.move2id[u])
    print(f"model played {len(hist)} plies in {time.time()-t0:.2f}s (~{(time.time()-t0)/max(1,len(hist))*1000:.0f} ms/move)")
    t0 = time.time(); mv = engine_move(chess.Board(), 2)
    print("engine L2 first move:", mv, f"({time.time()-t0:.2f}s)  <- if slow/approx, wire plan/engine_hook.py")
    sf = open_sf(); info = sf.analyse(chess.Board(), chess.engine.Limit(depth=8), multipv=3)
    print("stockfish ok:", [i["pv"][0].uci() for i in info]); sf.quit()
    print("ALL OK")


if __name__ == "__main__":
    main()
