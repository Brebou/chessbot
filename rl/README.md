# RL Framework (Phase 3)

Policy-gradient training of the fixed `ChessGPT` architecture against the
provided engine, without touching `chess_hw/`. Everything is seeded,
atomically written and resumable-by-rerun (checkpoints + `metrics.jsonl`).

## Quick start (GPU machine)

```bash
uv run train_rl.py --model-path base_model --out-dir results/rl/run1 \
    --levels 1 2 --games-per-iter 256 --group-size 4 \
    --lr 1e-5 --kl-coef 0.05 --replay-coef 0.5 \
    --eval-every 10 --eval-seeds 0 1 --eval-n-openings 25 \
    --eval-workers 16 --device cuda
```

Every `RLConfig` field is a `--kebab-case` CLI option (`uv run train_rl.py --help`).

## How one iteration works

1. **Rollouts** (`rl/rollout.py`): `games_per_iter` games vs
   `diag.fast_engine.FastEngine` (the exact eval opponent, levels 1–3),
   advanced in lockstep rounds — all model turns batched into one GPU
   sampling pass (`rl/policy.py`, last-position-only forward), engine replies
   in-process or in a `--engine-workers` process pool. Games follow
   `chess_hw.match.play_game` semantics: 250-ply draw adjudication, seeded
   random legal fallback when all sampled tokens are illegal.
2. **Rewards** (`rl/rewards.py`): per decision, dense material shaping
   `material_shaping × (m_{i+1} − m_i)` (material diff at the next model
   decision, i.e. after the engine replied), a fixed `--illegal-penalty` for
   sampled-illegal tokens, and `--result-reward × score` on the last
   decision. Returns are discounted (`--gamma`).
3. **Advantages**: GRPO-style group normalization — games sharing the same
   (opening, model color, engine level) form a group (`--group-size` games
   per group), returns are centered/scaled within the group; or a plain
   global baseline (`--baseline global`).
4. **Update** (`rl/losses.py`, `rl/train.py`): log-probs recomputed with a
   fresh forward pass; PPO clipped surrogate (`--ppo-clip`, 0 = plain
   REINFORCE) + KL penalty to the frozen reference policy (unbiased k3
   estimator on sampled tokens, `--kl-coef`) + Lichess replay CE
   (`--replay-coef`, anti-forgetting; skipped automatically if
   `data/lichess_train_2025-01_1M.jsonl.gz` is absent). AdamW, grad clip,
   `--inner-epochs` × `--minibatches`.
5. **Eval** (`--eval-every`): `chess_hw.match.evaluate_vs_engine` on each
   `--eval-seed` (multi-seed — the official seeds are private, never tune on
   one seed); best mean score is kept in `best/model.safetensors`.

## Run directory layout

```
results/rl/run1/
├── config.json          # exact hyperparameters of the run
├── metrics.jsonl        # one row per iteration (losses, scores, eval)
├── checkpoints/iter-XXXXXX/model.safetensors
├── best/model.safetensors        # submission candidate (best eval)
├── final/model.safetensors
└── summary.json
```

## Design notes / gotchas honored

- **Sampling vs eval semantics**: training samples from the full softmax
  (exploration); the deployed greedy top-3-first-legal behavior improves as
  the distribution sharpens. Illegal samples are penalized and replaced by
  the same seeded fallback the eval uses, so illegal mass gets a direct
  negative signal without corrupting the game distribution.
- **Histories fit n_ctx=256** (BOS + ≤255 moves); parity-preserving
  truncation is inherited from `FastProbePlayer`.
- **GRPO groups share the exact opening** — baselines compare identical
  starting conditions.
- **Determinism**: per-iteration seeds derive from `--seed`; the sampler uses
  a dedicated `torch.Generator`; writes are tmp + `os.replace`.
- **`chess_hw/` is never modified**; evaluation goes through its public API.
- Remember to `git checkout -- chess_hw/__pycache__` before committing after
  local runs.

## Tuning pointers

- Start with `--levels 1 2`, `--games-per-iter 256`, `--group-size 4`,
  `--lr 1e-5`, `--kl-coef 0.05`, `--replay-coef 0.5`, `--temperature 1.0`.
- Tune loss weights on the multi-seed eval score (`metrics.jsonl`,
  `eval_mean_score`), never on the training losses.
- If legality degrades in long games (watch `illegal_rate` and eval
  `random_move_rate`), raise `--replay-coef` or `--kl-coef`.
