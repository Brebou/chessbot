# ft — experiment ledger and status

Progress tracker for the post-training work. Updated after every run or
merged PR. Judged spec: the mix ratios, loss, eval protocol, and stop
criteria implemented under `ft/`. Refined acceptance rules and the
instruments they use are in `ft/eval/` (paired eval + frozen fixtures).

## Status at a glance

| Item | State |
|---|---|
| Submission candidate | `base_model` (paired score 0.231) |
| Fallback | `results/sft/run1/final` (0.210) |
| Checkpoint qualification rule | `balanced − 2σ > 0`; nothing qualified (8 post-training runs, ~30 evals) |
| Local suite | 127 passed / 7 skipped |
| Current stage | **Ladder terminal after run4 → final submission `base_model`; write-up pending** |

## How models are evaluated (3 instruments, all paired vs base)

1. **Paired engine games** (Elo-like): full games vs the official alpha-beta
   engine, levels 0–3, fixed openings/seeds both colors (~320 games);
   delta + cell-bootstrap sigma. FastEngine for L1–3 (move-identical, ~4×).
2. **Paired self-play proxy** (tournament proxy): 200 openings × 2 colors
   vs base. `balanced = mean(engine, proxy)` drives selection/plateau.
3. **Frozen fixtures** (single forward pass, sub-second): 500 verified
   mate-in-1 + 500/bin (mid/late/end) from real games. Noise bands:
   mate1 ±1.2 pts, captures ±5 pts, legality ±1 pt.

## Iteration-1 experiment matrix (all runs `--mix` from data/ft8h)

| Config | mate1 (base 4.8%) | win-cap late (base 64.1%) | engine Δ | proxy Δ | qualified |
|---|---|---|---|---|---|
| mix, hinge on (run1b, 2 ep; run1c 3.5 ep) | ~8.5% sat. | 58.7–60.4% | ~−0.03 | ~−0.047 | none (×8) |
| mix_w2 (`--bucket-weight mate=2.0`), hinge on (run2a) | 10.2%, rising at abort | 55.6–57.0% | ~−0.033 | −0.058 | none |
| mix, hinge off (`--lambda-contrast 0`) (run2b) | 8.4–8.8% | 58.2–62.5% | ~−0.03 | −0.015…−0.054 | none |
| mix_w2, hinge off (run2c / B2) | 10.0% sat. | 55.7–58.5% | −0.03…−0.048 | −0.031…−0.056 | none |
| mix_it2 = mix + 5.8% DAgger rows, 18,049 on-policy states, engine-labeled (run3) | 8.4–9.2% | 59.0–59.5% (not recovered) | −0.042 → **−0.0156** (closest-to-zero of the campaign, ~1.1σ) | −0.039…−0.034 | none |
| mix_it2b = mix + 15%-capped DAgger-2 (32,114 CE rows → normal + 20,827 on-policy contrastive pairs; 87,860 states, depth 10), from base (run4) | 7.4–9.6% | 56.5–59.0% (not recovered) | −0.033 → −0.048 (monotone decline; best eval −0.0297) | −0.030…−0.046 | none |

Robust conclusions (946k params):
- mate1 dose-responds to CE weight (4.8 → 10%) then saturates; the
  contrastive hinge is inert on fixtures and its removal changes nothing.
- Every variant of the off-policy elite mix costs ~4–6 pts of late-bin
  capture play and ~2–5% head-to-head score → checkpoint selection
  correctly refuses everything; off-policy SFT is exhausted as an Elo lever.
- Legality is banked in the base (98% global / 95% end on our fixtures),
  preserved under all runs; monitor via guards only, never optimize.
- On-policy model-negative pairs are actively harmful at this capacity:
  run4 vs run3 at the matched step-1152 eval is engine −0.0453 vs −0.0156,
  and run4's balanced score declines monotonically with training while
  `val_ce_contrastive` *improves* (3.66 → 3.30). The model fits its own
  visited states' negatives readily, but CE-pulling toward off-manifold
  SF-best at exactly the states its current policy produces outweighs the
  hone signal on ~1M params.
- The one robust real improvement of the campaign is mate-in-1 (4.8% →
  7.4–10.2% across all variants); it never converts into engine score.

## Decision rules (pre-registered)

- Checkpoints qualify only via `balanced − 2σ > 0`; fixtures are diagnostics.
- Fixtures deltas only count beyond their noise bands (see above).
- Note: contrastive rows already carry full CE toward the good move at
  weight 1.0 (`ft_loss` runs `weighted_ce` over the whole batch) — "punish
  must also build" is the implemented shape, no extra term needed.
- DAgger OOD detector is inert at the (pieces, material-balance)
  signature granularity (n_ood = 0 everywhere); harmless — the share cap
  never trims at 18k rows.
- Stage-2 (run3) started from `base_model`, *not* the failed SFT policies:
  no SFT checkpoint earned the chain, and re-exposure to the same mix
  doubles the drift the proxy metric penalizes.
- Escalation ladder if run3 is flat: (a) add CE-on-good to contrastive
  pairs + model-generated negatives from the same DAgger rollouts
  (“punish must also build”), (b) legal-masked CE, (c) soft targets from
  annotated `cp_second`. If none converts, submit `base_model`.
- Ladder terminal (recorded after run4): (a) was already half-implemented
  (CE-on-good ships in `ft_loss`); its new half, on-policy negatives,
  made the engine trend strictly worse than run3. (b) declined: legality
  is 97–100% top-1, no headroom. (c) declined: it reweights the same
  off-policy CE signal that every run shows degrades engine play.
  **Final submission: `base_model` (0.231).**

- run4 pre-flight crash: `rollout_positions`/`dagger_collect` wrapped the
  `progress` object as `show` instead of its `.show` method (fast path
  untested — CLI progress only); fixed with a stub-object regression test.

## Run log

| Run | Config | Outcome |
|---|---|---|
| sft/run1 (pre-ft) | SFT mix v1 | 0.210 -- kept as fallback |
| ft/run1 | mix, 2 ep | epoch bug (pre-fix), stale artifact, deletable |
| ft/run1b | mix, 3 ep, patience 3 | plateau abort, 4 flat evals |
| ft/run1c | mix, 7 ep, patience 7 | aborted at 3.5 ep; val CE turns over at den 3 |
| ft/run2a | mix_w2, patience 4 | mate1 10.2% rising; aborted by balanced plateau |
| ft/run2b (B1) | mix, no hinge | no change vs run1b |
| ft/run2c (B2) | mix_w2, no hinge | mate1 10% sat., captures not recovered |
| ft/run3 | DAgger from base (annotation pool) | 18,049 states collected+annotated (depth 10); 0 qualified; engine trend nearest zero of campaign; captures not recovered |
| ft/run4 | mix_it2b (DAgger-2 15% cap: CE→normal + on-policy pairs), from base, 3 ep | 1,728 steps, 6 evals; balanced declines −0.033 → −0.047; engine best −0.0297; strictly worse than run3 at matched step-1152; no qualification → ladder terminal, submit `base_model` |
