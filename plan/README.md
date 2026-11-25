# plan/ — scaled on-policy Stockfish distillation (+ search-based expert iteration)

Drop this folder at the repo root (next to `base_model/`), then run everything as `python -m plan.<script>`.
Needs: torch, transformers, safetensors, python-chess, Stockfish (`STOCKFISH_PATH` or `~/.local/bin/stockfish`).

**Written without access to your repo code and untested against it.** Two adapter points:
1. `common.load_model / load_vocab / save_checkpoint` (HF GPT2-style assumed).
2. `engine_hook.py` — wire the OFFICIAL level 1/2/3 engine. Without it, a slow approximate engine is used and
   every number (and beam.py especially) is about the wrong opponent.
`python -m plan.selfcheck` verifies all of this in 1 minute. Do it first.

## Why this differs from the previous campaign
- Each game gives a training signal at EVERY ply (soft Stockfish multipv targets), not one row per position:
  ~100x more signal per forward pass; 3-5M labelled positions instead of 300k.
- Real doses: lr 1e-4, 2-3 epochs, EMA, no/low KL. Select by a dense offline metric, not 240 noisy games.
- Draws count 0.5: check the `endings:` line of play.py (forfeits? 250-ply cap?) before anything else.

## Pipeline
```bash
# 0. sanity + diagnostics (loss taxonomy: checkmate / forfeit / cap / draw)
python -m plan.selfcheck --model base_model
python -m plan.play --model base_model --n-openings 100 --seed 1234 --save results/base.jsonl

# 1. round 1: collect on-policy games (use many openings; greedy play is deterministic, openings = diversity)
python -m plan.play --model base_model --opps L1 L2 L3 M:base_model --n-openings 3000 --seed 1  --save data/r1/games.jsonl
python -m plan.play --model base_model --opps L1 L2 L3 M:base_model --n-openings 150  --seed 99 --save data/heldout/games.jsonl
python -m plan.label --games data/r1/games.jsonl      --out data/r1/shards      --depth 8
python -m plan.label --games data/heldout/games.jsonl --out data/heldout/shards --depth 8 --opp-weight 0

# 2. baseline offline metrics, then train
python -m plan.offline_eval --model base_model --shards data/heldout/shards
python -m plan.train --data data/r1/shards --init base_model --out out/r1 --epochs 3 --lr 1e-4 --device cuda
python -m plan.offline_eval --model out/r1/last --shards data/heldout/shards    # blunder_rate must fall clearly

# 3. games, paired against base (same seed => same cells)
python -m plan.play --model out/r1/last --n-openings 300 --seed 1234 --baseline results/base.jsonl --save results/r1.jsonl
python -m plan.play --model out/r1/last --opps M:base_model --n-openings 300 --seed 5     # Elo proxy (score vs base)

# 4. DAgger round 2/3: collect with out/r1/last, label, train with --init out/r1/last (reuse r1 shards too:
#    --data data/r2/shards data/r1/shards)

# 5. expert iteration (needs official engine in engine_hook.py)
python -m plan.beam  --model out/r2/last --levels 2 3 --n-openings 1500 --out data/beam/wins.jsonl
python -m plan.label --games data/beam/wins.jsonl --out data/beam/shards --mode winmoves
python -m plan.train --data data/r2/shards data/beam/shards --init out/r2/last --out out/r3 --epochs 2 --lr 5e-5
```
Final: copy `out/<best>/model.safetensors` to the submission. Re-run your OFFICIAL eval (diag/run_eval.py)
for the real numbers; `play.py` is for development.

## Decisions
- Primary metric: held-out `blunder_rate`, `mean_cp_loss_lb`, `mate_hit` (offline_eval). If a properly scaled
  round (>=1M labelled plies, lr 1e-4) does NOT improve them, capacity really is the limit — only then fall back.
- Game confirm: 300 openings (σ≈0.006 paired). Submit best point estimate that is positive on BOTH the engine
  score and the vs-base proxy. Base scores 0 points; do not require 2σ.
- Knobs if offline metric improves but games don't: raise `--opp-weight 0` (only our-side positions),
  `--temp 30` (sharper targets), more DAgger rounds, add beam shards.
- Hold out openings: never collect train and eval with the same --seed (or the same book lines).
