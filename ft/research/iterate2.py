"""Iterated book-aligned corrections: does the loop compound or decay?

gen 0 = base; for gen g >= 1:
  1. play `(gen g-1 model) vs class engine` on a FIXED set of book openings
     (paired across generations) -> fresh state corpus each gen (never reused)
  2. SF-score -> corrections + 1:1 rehearsal, cap 20k
  3. fine-tune (kl 0.1, lr 5e-5, 1 epoch) -> gen g model
  4. paired engine eval (fixed seed) vs BASE + the previous gen model
Rows appended to ft/research/results.jsonl. Run it in the background and
check ft/research/results.jsonl for the per-generation trend.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ft.research.sweep import EVAL_LEVELS, build_mix, resolve_engine, train_candidate
from ft.research.arc2 import _select_oids, collect_book_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="base_model/model.safetensors")
    ap.add_argument("--anchor-path", default=None)
    ap.add_argument("--out-dir", default="results/it2")
    ap.add_argument("--n-openings", type=int, default=150)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--sf-depth", type=int, default=6)
    ap.add_argument("--sf-workers", type=int, default=4)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--kl", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--corpus-seed", type=str, default="it2a")
    ap.add_argument("--eval-openings", type=int, default=10)
    ap.add_argument("--selfplay-games", type=int, default=20)
    ap.add_argument("--eval-workers", type=int, default=2)
    ap.add_argument("--eval-seed", type=int, default=1234)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.anchor_path or args.model_path
    engine = resolve_engine(args.engine)
    oids = _select_oids(args.corpus_seed, args.n_openings)  # fixed across gens

    from ft.eval.paired_eval import paired_engine_eval, selfplay_paired

    policy = args.model_path
    for gen in range(1, args.generations + 1):
        t0 = time.monotonic()
        gdir = out_dir / f"gen{gen}"
        gdir.mkdir(parents=True, exist_ok=True)
        print(f"=== gen {gen}: corpus from {policy} ===", flush=True)
        raw_path = gdir / "corpus.jsonl"
        if raw_path.exists():
            from ft.common.serialize import jsonl_read
            corpus = list(jsonl_read(raw_path))
            print(f"corpus: loaded {len(corpus)} states from disk", flush=True)
        else:
            corpus = collect_book_corpus(policy, oids, args.levels, workers=1)
            merged: dict = {}
            for s in corpus:
                merged.setdefault(s["fen"], s)
            corpus = list(merged.values())
            scores_done = out_dir / f".scores_gen{gen}"
            if scores_done.exists():
                from ft.common.serialize import jsonl_read
                scored = {s["fen"]: s.get("score") for s in
                          jsonl_read(scores_done)}
                for s in corpus:
                    s["score"] = scored.get(s["fen"], {})
            else:
                from ft.sft.poc_loop import poc_score
                from ft.common.board_io import n_pieces_from_fen
                from ft.sft.poc_loop import poc_classify
                print(f"corpus: {len(corpus)} unique states; SF scoring...",
                      flush=True)
                scores = poc_score(
                    [(s["fen"], s["model_move"]) for s in corpus],
                    engine_path=engine, depth=args.sf_depth,
                    workers=args.sf_workers)
                n_corr = 0
                for s in corpus:
                    sc = scores.get(s["fen"], {})
                    s["score"] = sc
                    s["corrected"] = poc_classify(
                        sc.get("cp_best"), sc.get("cp_model"),
                        sc.get("mate_best"), sc.get("mate_model"),
                        n_pieces_from_fen(s["fen"]))
                    n_corr += 1 if s["corrected"] else 0
                print(f"corpus: {n_corr}/{len(corpus)} corrections",
                      flush=True)
                from ft.common.serialize import jsonl_write
                jsonl_write(scores_done, corpus)
            from ft.common.serialize import jsonl_write
            jsonl_write(raw_path, corpus)

        mix_path = out_dir / f"mix_gen{gen}.jsonl"
        if not mix_path.exists():
            n_rows, n_corr = build_mix(corpus, mix_path, rehearsal=1.0,
                                       seed=gen, cap=args.cap)
            print(f"mix: {n_rows} rows ({n_corr} corrections)", flush=True)

        done = train_candidate(
            mix_path, model_path=policy, base_path=base_path,
            out_dir=gdir / "model", epochs=args.epochs, lr=args.lr,
            kl=args.kl, batch=args.batch, seed=gen,
            eval_openings=args.eval_openings,
            selfplay_games=args.selfplay_games,
            eval_workers=args.eval_workers, records_cache={},
            device=args.device)
        mpath = done.get("final_model", str(gdir / "model" / "final" /
                                            "model.safetensors"))
        p = paired_engine_eval(mpath, base_path, levels=EVAL_LEVELS,
                               n_openings=args.eval_openings,
                               seed=args.eval_seed, workers=args.eval_workers,
                               n_bootstrap=1000, records_cache={})
        sl = selfplay_paired(mpath, base_path, n_games=args.selfplay_games,
                             seed=args.eval_seed, workers=args.eval_workers)
        row = {"run": f"it2/gen{gen}", "engine_delta": p["delta"],
               "sigma": p["sigma"], "per_level": p["per_level_delta"],
               "proxy_delta": sl["proxy_delta"],
               "balanced": (p["delta"] + sl["proxy_delta"]) / 2.0,
               "policy_vs_prev": None, "mix_rows": n_rows,
               "elapsed_s": round(time.monotonic() - t0)}
        # also compare against the previous generation model to detect decay
        if gen > 1:
            prev = out_dir / f"gen{gen-1}" / "model" / "final" / \
                "model.safetensors"
            p2 = paired_engine_eval(mpath, str(prev), levels=EVAL_LEVELS,
                                    n_openings=args.eval_openings,
                                    seed=args.eval_seed,
                                    workers=args.eval_workers,
                                    n_bootstrap=500, records_cache={})
            row["policy_vs_prev"] = p2["delta"]
        with open(Path("ft/research/results.jsonl"), "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"=== gen {gen} done: vs_base={row['engine_delta']:+.4f} "
              f"vs_prev={row['policy_vs_prev']} balanced={row['balanced']:+.4f} "
              f"({row['elapsed_s']}s) ===", flush=True)
        policy = mpath


if __name__ == "__main__":
    main()
