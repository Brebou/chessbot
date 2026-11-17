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

1. **Rollouts** (`rl/rollout.py`): `games_per_iter` games, mixed between two
   sources, advanced in lockstep rounds — all model turns batched into one
   GPU sampling pass (`rl/policy.py`, last-position-only forward), engine
   replies in-process or in a `--engine-workers` process pool. Games follow
   `chess_hw.match.play_game` semantics: 250-ply draw adjudication, seeded
   random legal fallback when all sampled tokens are illegal.
   - **Engine games** (`1 - selfplay_ratio`): vs `FastEngine` (the exact
     eval opponent, levels 1–3) — optimizes the engine score (grade 1).
   - **Self-play games** (`selfplay_ratio`, default **0.5**): the policy
     plays both colors — keeps the training distribution tournament-like.
2. **Rewards** (`rl/rewards.py`): per decision, dense material shaping
   `material_shaping × (m_{i+1} − m_i)` (material diff at the mover's next
   decision, i.e. after the opponent replied), a fixed `--illegal-penalty`
   for sampled-illegal tokens, and `--result-reward × score(color)` on the
   last decision of each color (in self-play, each side is credited with its
   own outcome). Returns are discounted (`--gamma`).
3. **Advantages**: GRPO-style group normalization — games sharing identical
   starting conditions ((opening, model color, engine level) for engine
   games, (opening,) for self-play) form a group (`--group-size` games per
   group), returns are centered/scaled within the group; or a plain global
   baseline (`--baseline global`).
4. **Update** (`rl/losses.py`, `rl/train.py`): log-probs recomputed with a
   fresh forward pass; PPO clipped surrogate (`--ppo-clip`, 0 = plain
   REINFORCE) + KL penalty to the frozen reference policy (unbiased k3
   estimator on sampled tokens, `--kl-coef`) + Lichess replay CE
   (`--replay-coef`, anti-forgetting; skipped automatically if
   `data/lichess_train_2025-01_1M.jsonl.gz` is absent). AdamW, grad clip,
   `--inner-epochs` × `--minibatches`.
5. **Eval** (`--eval-every`): two signals on each `--eval-seed` (multi-seed
   — the official seeds are private, never tune on one seed):
   - `eval_mean_score`: vs the engine (`chess_hw.match.evaluate_vs_engine`);
   - `eval_vs_ref_score`: **Elo proxy** — current policy vs the frozen
     reference model, round-robin style. This is the tournament-relevant
     signal.

   The best checkpoint is kept by `--best-criterion`: `engine`, `vs_ref`, or
   `balanced` (default: mean of both).

## ⚠️ The engine-vs-tournament trade-off (do not skip)

Experience from other students: **over-optimizing against the alpha-beta
engine raises the engine score but LOWERS round-robin Elo** — the policy
learns to exploit a depth-limited, PST-driven opponent instead of playing
generally strong chess against weak LLMs. The framework therefore:

- mixes **50% self-play** into the rollouts by default (`--selfplay-ratio`);
- tracks `eval_vs_ref_score` (round-robin proxy) next to the engine score;
- keeps the best checkpoint on the **balanced** criterion by default.

Operational rule while tuning: if `eval_mean_score` goes up while
`eval_vs_ref_score` goes down → raise `--selfplay-ratio` (toward 0.7–1.0),
lower `--levels`, and/or re-weight toward replay/SFT data. Only accept a
checkpoint whose *balanced* score improved.

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
  `--selfplay-ratio 0.5`, `--lr 1e-5`, `--kl-coef 0.05`, `--replay-coef 0.5`,
  `--temperature 1.0`, `--best-criterion balanced`.
- Tune loss weights on the **balanced multi-seed eval score**
  (`best_criterion_score` in `metrics.jsonl`), never on the training losses.
- If legality degrades in long games (watch `illegal_rate` and eval
  `random_move_rate`), raise `--replay-coef` or `--kl-coef`.
- If `eval_vs_ref_score` stagnates or drops while the engine score climbs,
  you are over-fitting the alpha-beta opponent: raise `--selfplay-ratio`,
  drop `--levels` to 1, and re-check.
