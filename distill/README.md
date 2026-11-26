# Engine distillation pipeline

Everything here uses only the fixed `chess_hw` API. Training on the provided
engine is explicitly allowed by the task rules.

Idea: the opponents are levels 1-3 of `chess_hw.engine`. A model that imitates
the level-3 teacher's moves scores ~0.5 against L3 (deterministic mirror games
tend to draw) and well above that against L1/L2, which maximizes
`P = (S1 + S2 + S3) / 3`. Distilling the actual opponent beats Stockfish
labels: the target policy is exactly the deployment policy, and labels are
free for plies the teacher itself played (levels 1-4 are deterministic).

## Step 1 — offline distillation from teacher self-play

```bash
# ~26 s per 100-ply game per core; 3000 games ~ 1.5 h on 18 cores
python distill/gen_labels.py --n_games 3000 --seed 0 --out work/games_sft.pkl

python distill/train_sft.py --games work/games_sft.pkl --out work/distill_r1 \
    --epochs 6 --lr 2e-4 --bs 32            # add --data <prefix> for Lichess replay (.bin/.idx)

# paired held-out eval, same openings for both models
python distill/eval_paired.py --models base_model work/distill_r1 \
    --seed 777 --n_openings 100 --levels 1 2 3 --workers 8 --out work/eval_r1.json
```

## Round 2b — GRPO (optional, on top of the distilled policy)

```bash
python distill/grpo.py --init work/distill_r2 --out work/grpo \
    --iters 15 --games_per_iter 128 --group 8 --opps 1 2 --workers 15 --lr 1e-5
```

Samples legal moves at `--temp` from the current policy (cheap: ~2 ms/move),
plays G games per (opening, color, opponent) group, and takes one gradient
step per iteration on group-normalized game rewards with a KL anchor to the
base model. Keep opponents at L1/L2 (L3 rollouts are ~15x slower). Expect a
sharpening effect on top of distillation, not a replacement for it: the
signal is sparse (win/draw/loss) and transfer from L1/L2 opponents to L3 is
partial. Watch the per-iteration `reward L1/L2` trend; snapshots are saved
every `--save_every` iterations.

## Round 2 — DAgger (fixes distribution shift + forgetting)

```bash
python distill/dagger.py --model work/distill_r1 --prev work/games_sft.pkl \
    --games 2000 --seed 1 --opps 2 3 --teacher 3 --out work/distill_r2 \
    --epochs 6 --lr 2e-4
```

Plays the current model vs L2/L3, labels the model's plies with the L3
teacher, aggregates with all previous data, and retrains **from base**.
Repeat with `--model work/distill_r2 --prev work/games_sft.pkl work/distill_r2_games.pkl --seed 2 ...`.

## Capacity probe

```bash
python distill/agreement.py --model work/distill_r1 --n_games 128 --workers 8
```

Held-out L3 positions: `top1` (argmax == teacher), `top3` (teacher move in
top-3 tokens — the fixed 3-attempt inference then plays it if legal),
`legal_top3`. If `top1` plateaus at a low value while you add data, the
1M-parameter capacity binds; stop scaling data and focus on legality/DAgger.

## Progress and error visibility

- All collection scripts (`gen_labels.py`, `dagger.py`, `agreement.py`,
  `eval_paired.py`) print a progress line every `--log_every` items with
  elapsed time, ETA and running stats (labeled positions, running score per
  opponent level, fallback counts). `--verbose` prints one line per game.
- `train_sft.py` prints loss components every `--log_every` steps (default 20)
  plus per-epoch dev agreement; per-file game/label counts are printed at load.
- Worker crashes are re-raised with the exact task/game attached
  (`worker failed on task (...)`), instead of a bare traceback.

## What to watch

- `dev top1` during training: the single best capacity/progress signal.
- `random-move rate` in `eval_paired.py`: must go to ~0; a fallback rate above
  ~1% means the legality loss weight (`--lam_legal`) or replay (`--lam_rep`)
  needs raising.
- Never compare scores across different openings: always use `eval_paired.py`
  (paired seeds).

## Files

- `common.py` — data format, batching, replay/target building
- `gen_labels.py` — teacher self-play labeling (step 1)
- `train_sft.py` — SFT training loop (legality + CE, optional replay, best-dev checkpoint)
- `dagger.py` — on-policy collection + relabel + retrain from base
- `eval_paired.py` — paired held-out evaluation (P metric)
- `agreement.py` — teacher-agreement capacity probe
